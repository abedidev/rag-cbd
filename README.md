# Evidence-Informed Guidance on Cannabidiol Use in Older Adults

**Development and Evaluation of Retrieval-Augmented Large Language Models**

Ali Abedi, Charlene H. Chu, Shehroz S. Khan

University of Toronto | KITE Research Institute

---

## About

This repository contains the code and data for a retrieval-augmented LLM framework that generates safe, personalized CBD educational guidance for older adults, including those with cognitive impairment. The system integrates structured prompt engineering with evidence retrieval from 32 curated clinical CBD resources and was evaluated across 64 diverse older adult scenarios using an automated, annotation-free evaluation pipeline.

## Paper

> A. Abedi, C. H. Chu, and S. S. Khan, "Evidence-Informed Guidance on Cannabidiol Use in Older Adults: Development and Evaluation of Retrieval-Augmented Large Language Models," 2025.

**[Read the full paper pre-print](https://doi.org/10.13140/RG.2.2.26646.82246)**

## Key Findings

- Retrieval-augmented models recommended starting doses of 2-5 mg, matching clinical guidelines, while standalone models often recommended 3-5x higher doses
- The ensemble RAG configuration (two LLMs + tiebreaker judge) produced the most cautious and clinically aligned outputs
- RAG models adjusted guidance for vulnerable profiles (older age, cognitive impairment, organ impairment) with up to 97% alignment, compared to 0-12% for some standalone models
- Claude Sonnet 4.5 declined to generate CBD content entirely, reflecting a conservative safety-by-design approach

## Models Evaluated

| Model | Type |
|---|---|
| OpenAI GPT 5.1 | Standalone + RAG |
| Google Gemini 2.5 Pro | Standalone + RAG |
| Mistral AI Medium 3 | Standalone |
| Anthropic Claude Sonnet 4.5 | Standalone (declined) |
| xAI Grok 4 | Standalone |
| DeepSeek V3.2-Exp | Standalone |
| Ensemble RAG (GPT 5.1 + Gemini 2.5 Pro + GPT 5.1 tiebreaker) | Ensemble |

## Evaluation Methods

Three automated, annotation-free evaluation methods were used:

1. **Statistical Consensus Evaluation** - measures deviation from the collective model distribution using z-scores
2. **Feature-Aligned Directional Evaluation** - checks whether models adjust dosage in clinically expected directions based on risk factors
3. **LLM-as-a-Judge Rubric Evaluation** - scores outputs on relevance, grounding, safety, structure, and clarity (0-5 scale)


## Acknowledgements

This research was funded by the Alzheimer's Association through a grant awarded to Charlene H. Chu.

## Citation

```bibtex
@article{abedi2025cbd,
  title={Evidence-Informed Guidance on Cannabidiol Use in Older Adults: Development and Evaluation of Retrieval-Augmented Large Language Models},
  author={Abedi, Ali and Chu, Charlene H. and Khan, Shehroz S.},
  year={2025}
}
```

## Contact

Ali Abedi - ali.abedi@uhn.ca
