import os
import json
from pathlib import Path

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --------------------------------------------------------------------
# Config and paths
# --------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GEMINI_MODEL_NAME = "gemini-2.5-pro"   # Gemini generator
GPT_MODEL_NAME = "gpt-5.1"             # GPT 5 generator
JUDGE_MODEL_NAME = "gpt-5.1"           # GPT 5 judge (can be same or different)

INPUT_PATH = Path("")
OUTPUT_PATH = Path("")
SYSTEM_PROMPT_PATH = Path("system-prompt-rag.txt")

CBD_PDF_FOLDER = Path("")
FAISS_INDEX_DIR = Path("")

# --------------------------------------------------------------------
# Helper: build retrieval query from intake JSON
# --------------------------------------------------------------------
def build_cbd_query(intake_json_str: str) -> str:
    intake = json.loads(intake_json_str)

    parts = []
    parts.append("Educational CBD dosing and safety guidance for older adults.")

    age = intake.get("age")
    if age:
        parts.append(f"Patient age {age} years.")
        if age >= 60:
            parts.append("Older adult population.")
    sex = intake.get("sex")
    if sex:
        parts.append(f"Sex: {sex}.")

    height = intake.get("height_cm")
    weight = intake.get("weight_kg")
    if height or weight:
        hw_bits = []
        if height:
            hw_bits.append(f"height {height} cm")
        if weight:
            hw_bits.append(f"weight {weight} kg")
        parts.append("Body size: " + ", ".join(hw_bits) + ".")

    goal = intake.get("goal")
    if goal:
        parts.append(f"Treatment goal: {goal}.")

    form = intake.get("form_preference")
    if form:
        parts.append(f"Preferred form: {form.replace('_', ' ')}.")

    ch = intake.get("cannabis_history", {}) or {}
    if ch.get("cbd_naive"):
        parts.append("CBD naive.")
    if ch.get("thc_sensitive"):
        parts.append("THC sensitive.")

    hepatic = intake.get("hepatic_function")
    if hepatic:
        parts.append(f"Hepatic function: {hepatic}.")
    renal = intake.get("renal_function")
    if renal:
        parts.append(f"Renal function: {renal}.")

    comorbid = intake.get("comorbidities") or []
    if comorbid:
        parts.append("Comorbidities: " + ", ".join(comorbid) + ".")

    meds = intake.get("meds") or []
    if meds:
        parts.append("Concomitant medications: " + ", ".join(meds) + ".")

    if intake.get("pregnancy"):
        parts.append("Pregnancy present.")

    parts.append(
        "Retrieve conservative CBD starting doses, titration schedules, "
        "monitoring recommendations, and situations where THC should be avoided."
    )

    return " ".join(parts)

# --------------------------------------------------------------------
# Helper: build or load FAISS
# --------------------------------------------------------------------
def build_vectorstore_from_pdfs(pdf_folder: Path, embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    if not pdf_folder.exists():
        raise RuntimeError(f"CBD guideline folder does not exist: {pdf_folder}")

    print(f"Loading CBD guideline PDFs from {pdf_folder} ...")
    loader = DirectoryLoader(
        str(pdf_folder),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    docs = loader.load()

    for doc in docs:
        source = Path(doc.metadata.get("source", "unknown"))
        doc.metadata["doc_id"] = source.name
        doc.metadata["doc_title"] = source.stem

    print(f"Loaded {len(docs)} documents. Splitting into chunks ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("FAISS vector store built.")
    return vectorstore


def get_or_build_vectorstore() -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GEMINI_API_KEY,
    )

    if FAISS_INDEX_DIR.exists():
        print(f"Loading existing FAISS index from {FAISS_INDEX_DIR} ...")
        vectorstore = FAISS.load_local(
            folder_path=str(FAISS_INDEX_DIR),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        print("FAISS index loaded.")
        return vectorstore

    print("No existing FAISS index found. Building a new one ...")
    vectorstore = build_vectorstore_from_pdfs(CBD_PDF_FOLDER, embeddings)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(folder_path=str(FAISS_INDEX_DIR))
    print(f"FAISS index saved to {FAISS_INDEX_DIR}.")
    return vectorstore

# --------------------------------------------------------------------
# Judge system prompt
# --------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """
You are a CBD education safety judge.

You receive:
- The original system instructions that define the JSON schema and safety rules.
- The intake JSON for a single scenario.
- The retrieved reference context from CBD guideline documents.
- Two candidate JSON outputs, labeled CANDIDATE_A and CANDIDATE_B.

Your task:
- Validate both candidates against the schema and safety rules.
- Prefer answers that are conservative, internally consistent, and well grounded in the reference context.
- Prefer evidence entries that correspond to higher similarity scores when both candidates are otherwise comparable.
- You may choose one candidate as is, or merge the best parts of both.
- If both candidates contain errors, correct them.

Output:
- Return a single final JSON object that strictly follows the schema described in the system instructions.
- Do not include any comments or extra text outside the JSON.
"""


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    if GEMINI_API_KEY is None:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment")
    if OPENAI_API_KEY is None:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    # Generators
    gemini_llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        api_key=GEMINI_API_KEY,
        temperature=0.0,
    )

    gpt_llm = ChatOpenAI(
        model=GPT_MODEL_NAME,
        temperature=0.0,
    )

    # Judge (GPT 5)
    judge_llm = ChatOpenAI(
        model=JUDGE_MODEL_NAME,
        temperature=0.0,
    )

    vectorstore = get_or_build_vectorstore()

    # Loop over intake files
    for in_path in sorted(INPUT_PATH.iterdir()):
        if not in_path.is_file():
            continue

        intake_text = in_path.read_text(encoding="utf-8").strip()
        if not intake_text:
            print(f"Skipping empty file: {in_path.name}")
            continue

        print(f"Processing {in_path.name} ...")

        # Retrieval
        query = build_cbd_query(intake_text)
        # retrieved_docs = vectorstore.similarity_search(query, k=3)
        retrieved_docs = vectorstore.similarity_search_with_relevance_scores(query, k=3)

        context_parts = []
        for i, (d, score) in enumerate(retrieved_docs, start=1):
            doc_id = d.metadata.get("doc_id", f"doc_{i}")
            doc_title = d.metadata.get("doc_title", "unknown_title")
            excerpt = d.page_content.strip()

            # Optionally store score in metadata too, if you ever need it later
            d.metadata["similarity_score"] = float(score)

            context_parts.append(
                f"[DOC_ID: {doc_id}]\n"
                f"[DOC_TITLE: {doc_title}]\n"
                f"[SCORE: {score:.3f}]\n"
                f"[EXCERPT]\n{excerpt}\n"
            )

        context_text = "\n\n".join(context_parts) if context_parts else "No relevant context was retrieved."

        # Shared human content for both generators
        human_content = f"""Reference context from external CBD resources:

{context_text}

Intake JSON:

{intake_text}
"""

        gen_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ]

        # Candidate A: Gemini
        gemini_resp = gemini_llm.invoke(gen_messages)
        candidate_a = gemini_resp.content

        # Candidate B: GPT 5
        gpt_resp = gpt_llm.invoke(gen_messages)
        candidate_b = gpt_resp.content

        # Save individual model outputs:
        # input_stem + "-" + model_name + original suffix
        stem = in_path.stem
        suffix = in_path.suffix

        gemini_out_path = OUTPUT_PATH / f"{stem}-{GEMINI_MODEL_NAME}{suffix}"
        gpt_out_path = OUTPUT_PATH / f"{stem}-{GPT_MODEL_NAME}{suffix}"

        gemini_out_path.write_text(candidate_a, encoding="utf-8")
        gpt_out_path.write_text(candidate_b, encoding="utf-8")

        # Judge step
        judge_human = f"""System instructions for the target JSON schema and safety rules:

{system_prompt}

Reference context:

{context_text}

Intake JSON:

{intake_text}

CANDIDATE_A:

{candidate_a}

CANDIDATE_B:

{candidate_b}

Choose the better candidate or merge them, fix any issues, and output a single final JSON that strictly follows the schema.
"""

        judge_messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=judge_human),
        ]

        final_resp = judge_llm.invoke(judge_messages)
        final_text = final_resp.content

        # Final judged output: original file name in output folder
        final_out_path = OUTPUT_PATH / in_path.name
        final_out_path.write_text(final_text, encoding="utf-8")

        print(f"  -> wrote {gemini_out_path}")
        print(f"  -> wrote {gpt_out_path}")
        print(f"  -> wrote {final_out_path} (judged final)")

if __name__ == "__main__":
    main()
