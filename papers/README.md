# Downloaded Papers

29 PDFs. The first ten are the papers named in the research specification; all ten were read in full and have detailed notes in `notes/<arxiv_id>_notes.md`. The other 19 were found by search and screened by abstract only.

`pages/` holds 3-page PDF chunks of the ten specified papers (made with the PDF chunker), with a manifest per paper.

## Specified papers (read in full)

1. [Can You Keep a Secret? Involuntary Information Leakage in Language Model Writing](2605.10794_can_you_keep_a_secret_involuntary_leakage.pdf)
   - Authors: Ari Holtzman, Peter West. Year: 2026. arXiv: 2605.10794
   - Why relevant: the paradigm we extend. Secret word + story, measured by 2AFC and free-response guessing. Leakage up to 79%; Gemma 3 12B at 83%.
2. [Towards eliciting latent knowledge from LLMs with mechanistic interpretability](2505.14352_eliciting_latent_knowledge_taboo.pdf)
   - Authors: Cywiński, Ryd, Rajamanoharan, Nanda. Year: 2025. arXiv: 2505.14352
   - Why relevant: Taboo models on Gemma 2 9B; logit lens and SAE readout of a secret word at layer 32.
3. [Eliciting Secret Knowledge from Language Models](2510.01070_eliciting_secret_knowledge.pdf)
   - Authors: Cywiński, Ryd, Wang, Rajamanoharan, Nanda, Conmy, Marks. Year: 2025. arXiv: 2510.01070
   - Why relevant: white-box readouts (logit lens, token-embedding similarity, SAE) and judge prompts; base-model control.
4. [LLMs Can't Play Hangman: On the Necessity of a Private Working Memory for Language Agents](2601.06973_llms_cant_play_hangman_private_memory.pdf)
   - Authors: Baldelli, Parviz, Zouaq, Chandar. Year: 2026. arXiv: 2601.06973
   - Why relevant: externalized private state fixes consistency; evidence on forcing versus offering a scratchpad.
5. [Probing the Lack of Stable Internal Beliefs in LLMs](2603.25187_lack_of_stable_internal_beliefs.pdf)
   - Authors: Luo, Xu, Lu, Yuan, Yao. Year: 2026. arXiv: 2603.25187
   - Why relevant: implicit goals held only "internally" drift across turns; fork-and-probe protocol.
6. [Don't Think of the White Bear: Ironic Negation in Transformer Models Under Cognitive Load](2511.12381_white_bear_ironic_negation.pdf)
   - Authors: Mann, Saxena, Tandon, Sun, Toteja, Zhu. Year: 2025. arXiv: 2511.12381
   - Why relevant: "do not mention X" primes X; semantic distractors strengthen rebound.
7. [Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory](2310.17884_confaide_can_llms_keep_a_secret.pdf)
   - Authors: Mireshghallah, Kim, Zhou, Tsvetkov, Sap, Shokri, Choi. Year: 2023 (ICLR 2024). arXiv: 2310.17884
   - Why relevant: ConfAIde benchmark; chain of thought did not reduce leakage.
8. [Emergent Response Planning in LLMs](2502.06258_emergent_response_planning.pdf)
   - Authors: Dong, Zhou, Liu, Yang, Lu. Year: 2025. arXiv: 2502.06258
   - Why relevant: probes show prompt representations encode future story content; position-sweep template.
9. [Do Language Models Plan Ahead for Future Tokens?](2404.00859_do_lms_plan_ahead_future_tokens.pdf)
   - Authors: Wu, Morris, Levine. Year: 2024 (COLM). arXiv: 2404.00859
   - Why relevant: pre-caching versus breadcrumbs; frames what an external plan could and could not offload.
10. [Creating Suspenseful Stories: Iterative Planning with Large Language Models](2402.17119_creating_suspenseful_stories.pdf)
    - Authors: Xie, Riedl. Year: 2024 (EACL). arXiv: 2402.17119
    - Why relevant: plan-then-write ladder with verbatim prompts, including "withhold the reason" instructions.

## Additional papers (abstract-level screening)

Semantic leakage and secrecy:

11. [Does Liking Yellow Imply Driving a School Bus? Semantic Leakage in Language Models](2408.06518_semantic_leakage_yellow_school_bus.pdf): Gonen et al., NAACL 2025. Defines semantic leakage of irrelevant prompt content.
12. [Scaling Down Semantic Leakage](2501.06638_scaling_down_semantic_leakage.pdf): Smilga, 2025. Qwen2.5 0.5B–7B; smaller models leak less.
13. [Do LLMs Strategically Reveal, Conceal, and Infer Information? The Chameleon Game](2501.19398_chameleon_game_reveal_conceal.pdf): Karabag et al., 2025.
14. [Suppressing Pink Elephants with Direct Principle Feedback](2402.07896_pink_elephants_direct_principle_feedback.pdf): Castricato et al., 2024. "Avoid entity X" as a controllability problem.
15. [LLM Probability Concentration: How Alignment Shrinks the Generative Horizon](2506.17871_llm_probability_concentration.pdf): Yang, Li, Holtzman. Basis of the "entropy budget" argument.
16. [Auditing Language Models for Hidden Objectives](2503.10965_auditing_hidden_objectives.pdf): Marks et al., 2025.

Planning in hidden states:

17. [Future Lens](2311.04897_future_lens.pdf): Pal et al., 2023. Single hidden states predict tokens two or more steps ahead.
18. [Unlocking the Future: Look-Ahead Planning Mechanistic Interpretability](2406.16033_lookahead_planning_mech_interp.pdf): Men et al., 2024.
19. [Extracting Paragraphs from LLM Token Activations](2409.06328_extracting_paragraphs_from_activations.pdf): Pochinkov et al., 2024. The "\n\n" token carries next-paragraph content.

Steering, ablation and SAEs:

20. [Refusal in Language Models Is Mediated by a Single Direction](2406.11717_refusal_single_direction.pdf): Arditi et al., 2024. Difference-of-means direction and directional ablation.
21. [Steering Language Models with Activation Engineering](2308.10248_activation_addition_steering.pdf): Turner et al., 2023.
22. [Steering Llama 2 via Contrastive Activation Addition](2312.06681_contrastive_activation_addition.pdf): Panickssery et al., 2023.
23. [LEACE: Perfect linear concept erasure in closed form](2306.03819_leace_concept_erasure.pdf): Belrose et al., 2023.
24. [Gemma Scope](2408.05147_gemma_scope.pdf): Lieberum et al., 2024. Open SAEs for Gemma 2.

Story generation with plans:

25. [Plan-And-Write](1811.05701_plan_and_write.pdf): Yao et al., 2019.
26. [Re3: Generating Longer Stories With Recursive Reprompting and Revision](2210.06774_re3_recursive_reprompting_revision.pdf): Yang et al., 2022.
27. [DOC: Improving Long Story Coherence With Detailed Outline Control](2212.10077_doc_detailed_outline_control.pdf): Yang et al., 2023.
28. [Co-Writing Screenplays and Theatre Scripts with Language Models (Dramatron)](2209.14958_dramatron_cowriting.pdf): Mirowski et al., 2022.
29. [Are Large Language Models Capable of Generating Human-Level Narratives?](2407.13248_llm_human_level_narratives.pdf): Tian et al., 2024. LLM stories lack suspense; turning-point annotations.
