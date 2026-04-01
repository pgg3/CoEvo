"""CoEvoSummarizer: embedding-based idea pool management.

Ported from methods/coevo/coevosummarizer.py, adapted to work with
Solution objects and CoEvoPromptBuilder.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from evotoolkit.tools import HttpsApi
    from evotoolkit.core import Solution

    from .prompts.coevo_prompts import CoEvoPromptBuilder


class CoEvoSummarizer:
    """Embedding-based idea pool for the CoEvo summarization mechanism.

    Parameters
    ----------
    prompt_builder:
        CoEvoPromptBuilder used to build summarizer prompts.
    llm:
        evotoolkit HttpsApi instance (returns ``(response_str, usage)``).
    pool_size:
        Maximum number of ideas to keep in the pool.
    num_idea_to_return:
        How many ideas to return per ``select_inspirations`` call.
    cluster_summary:
        Whether to use DBSCAN clustering for diversity.
    tokenizer_path:
        Path to a HuggingFace tokenizer/model for sentence embeddings.
    """

    def __init__(
        self,
        prompt_builder: CoEvoPromptBuilder,
        llm: HttpsApi,
        *,
        pool_size: int = 100,
        num_idea_to_return: int = 5,
        cluster_summary: bool = True,
        tokenizer_path: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.prompt_builder = prompt_builder
        self.llm = llm
        self.pool_size = pool_size
        self.num_idea_to_return = num_idea_to_return
        self.cluster_summary = cluster_summary

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer_inst = AutoTokenizer.from_pretrained(tokenizer_path)
        self.embedding_inst = AutoModelForCausalLM.from_pretrained(tokenizer_path).to(self.device)

        self.idea_pool: list[dict] = []
        self.embedding_list: list = []

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def load_summary(self, summary_content: list[dict]) -> None:
        self.idea_pool = summary_content
        if self.idea_pool:
            self.embedding_list = [
                self._get_sentence_embedding(json.dumps(s)) for s in summary_content
            ]

    # ------------------------------------------------------------------
    # Inspiration selection
    # ------------------------------------------------------------------

    def select_inspirations(self, current_inspiration_list: list[dict] | None = None) -> list[dict]:
        if not self.embedding_list:
            return []

        embeddings_array, cluster_col, scaler = self._analyze_cluster()

        if current_inspiration_list is not None:
            total_indices: list[int] = []
            for idea in current_inspiration_list:
                simplified = {"Name": idea["Name"], "Definition": idea["Definition"]}
                emb = self._get_sentence_embedding(json.dumps(simplified))
                indices = self._find_top_similar(embeddings_array, cluster_col, scaler, emb)
                total_indices.extend(indices)
            total_indices = list(set(total_indices))
        else:
            total_indices = self._find_top_similar(embeddings_array, cluster_col, scaler)

        pool_subset = [self.idea_pool[i] for i in total_indices]
        rand_idx = np.random.permutation(len(pool_subset))
        rand_idx = rand_idx[: self.num_idea_to_return]
        return [pool_subset[i] for i in rand_idx]

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def summarize_indiv(self, chain: list[Solution]) -> None:
        prompt = self.prompt_builder.get_summarizer_prompt_single(chain, self.idea_pool)
        self._prompt_parse_add(prompt)

    def summarize_offspring(
        self,
        parents: list[list[Solution]],
        offspring: list[list[Solution]],
    ) -> None:
        prompt = self.prompt_builder.get_summarizer_prompt_offspring(parents, offspring, self.idea_pool)
        self._prompt_parse_add(prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prompt_parse_add(self, prompt_content: str) -> None:
        parsed, response = self._prompt_till_valid(prompt_content)
        if parsed is None:
            return
        for new_idea in parsed:
            entry = {
                "Name": new_idea["Name"],
                "Definition": new_idea["Definition"],
                "Example": new_idea["Example"],
            }
            self.embedding_list.append(self._get_sentence_embedding(json.dumps(entry)))
            self.idea_pool.append(entry)

        if len(self.embedding_list) > self.pool_size:
            self.embedding_list = self.embedding_list[-self.pool_size :]
            self.idea_pool = self.idea_pool[-self.pool_size :]

    def _prompt_till_valid(self, prompt_content: str) -> tuple[list | None, str]:
        n_retry = 0
        while True:
            response_str, _usage = self.llm.get_response(prompt_content)
            parsed, success = self._parse_response(response_str)
            if success:
                return parsed, response_str
            n_retry += 1
            if n_retry > 3:
                return None, response_str

    def _parse_response(self, response_str: str) -> tuple[list | None, bool]:
        try:
            idea_heading = r"(?:new|New|NEW)\s*(?:idea|Idea|IDEA)[sS]?\s*:?"
            inspirations_pattern = re.compile(
                r"##\s*" + idea_heading + r"\s*(.*?)##\s*",
                re.DOTALL,
            )
            match = inspirations_pattern.search(response_str)
            if not match:
                return None, False

            inspirations_text = match.group(1).strip()
            inspiration_pattern = re.compile(
                r"(?:reasoning|Reasoning|REASONING|reason|Reason|REASON)[sS]?\s*:?\s*(.*?)\s*\n.*?"
                r"(?:name|Name|NAME)\s*:?\s*(.*?)\s*\n.*?"
                r"(?:definition|Definition|DEFINITION)\s*:?\s*(.*?)\s*\n.*?"
                r"(?:example|Example|EXAMPLE)[sS]?\s*:?\s*(.*?)\s*-*\s*(?=idea|Idea|IDEA|$)",
                re.DOTALL,
            )
            parsed_inspirations = []
            for m in inspiration_pattern.findall(inspirations_text):
                parsed_inspirations.append({
                    "Reasoning": m[0].strip(),
                    "Name": m[1].strip(),
                    "Definition": m[2].strip(),
                    "Example": m[3].strip(),
                })
        except Exception:
            return None, False

        return parsed_inspirations, True

    def _analyze_cluster(self):
        import torch
        from scipy.spatial.distance import cdist
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        paragraph_embeddings = torch.stack(self.embedding_list)
        embeddings_array = paragraph_embeddings.detach().cpu().numpy()
        scaler = StandardScaler()
        embeddings_array = scaler.fit_transform(embeddings_array)

        if self.cluster_summary and len(self.embedding_list) > self.num_idea_to_return * 2:
            min_sample = max(1, len(self.embedding_list) // self.num_idea_to_return // 2)
            dbscan = DBSCAN(eps=0.6, min_samples=min_sample, metric="cosine")
            clusters = dbscan.fit_predict(embeddings_array)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            if n_clusters == 0:
                cluster_col = np.arange(len(self.embedding_list)).reshape(-1, 1)
            else:
                cluster_col = clusters.reshape(-1, 1)
        else:
            cluster_col = np.arange(len(self.embedding_list)).reshape(-1, 1)

        return embeddings_array, cluster_col, scaler

    def _find_top_similar(
        self,
        embeddings_array,
        cluster_col,
        scaler,
        current_embedding=None,
    ) -> list[int]:
        from scipy.spatial.distance import cdist

        if current_embedding is not None:
            emb_np = current_embedding.detach().cpu().numpy()
            standardized = scaler.transform(emb_np.reshape(1, -1))
            distances = cdist(standardized, embeddings_array, "euclidean")
            sorted_indices = np.argsort(distances[0])
        else:
            sorted_indices = np.arange(len(embeddings_array))

        top_indices: list[int] = []
        visited_clusters: list = []
        for index in sorted_indices:
            cluster = cluster_col[index, 0]
            if cluster != -1 and cluster not in visited_clusters:
                top_indices.append(int(index))
                visited_clusters.append(cluster)
            elif cluster == -1:
                top_indices.append(int(index))

        return top_indices

    def _get_sentence_embedding(self, sentence: str):
        import torch

        with torch.no_grad():
            t_input = self.tokenizer_inst(sentence, return_tensors="pt")
            t_input_dict = {k: v.to(self.device) for k, v in t_input.items()}
            outputs = self.embedding_inst(**t_input_dict, output_hidden_states=True)
            embedding = outputs.hidden_states[-1][0].mean(dim=0)
        return embedding
