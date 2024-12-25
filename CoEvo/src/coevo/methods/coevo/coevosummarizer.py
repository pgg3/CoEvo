import re
import json

import sklearn.metrics.pairwise
import torch
import numpy as np
from scipy.special import ellip_normal

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from .coevoparas import CoEvoParas
from .coevoprompts import CoEvoPrompt



class CoEvoSummarizer:
    def __init__(self, task_info_inst, paras: CoEvoParas, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer_inst = AutoTokenizer.from_pretrained(paras.tokenizer_path)
        self.embedding_inst = AutoModelForCausalLM.from_pretrained(paras.tokenizer_path).to(self.device)

        self.task_info_inst = task_info_inst
        self.paras = paras

        self.idea_pool = []
        self.embedding_list = []

    def load_summary(self, summary_content:list[dict]):
        self.idea_pool = summary_content
        if len(self.idea_pool) != 0:
            # Only leverage name and definition for clustering
            self.embedding_list = [
                self._get_sentence_embedding(json.dumps(each_sum)) for each_sum in summary_content
            ]

    def analyze_cluster(self):
        # Now self.embedding_list is a list with N tensors with the shape of (768,)
        paragraph_embeddings = torch.stack(self.embedding_list)
        embeddings_array = paragraph_embeddings.detach().cpu().numpy()
        scaler = StandardScaler()
        embeddings_array = scaler.fit_transform(embeddings_array)  # (n, 768)

        if self.paras.cluster_summary and len(self.embedding_list) > self.paras.num_idea_to_return*2 :
            min_sample = int(len(self.embedding_list) // self.paras.num_idea_to_return//2)
            if min_sample == 0:
                min_sample = 1
            dbscan = DBSCAN(eps=0.6, min_samples=min_sample, metric="cosine")
            clusters = dbscan.fit_predict(embeddings_array)

            n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
            if n_clusters_ == 0:
                cluster = np.arange(len(self.embedding_list))
                cluster_col = cluster.reshape(-1, 1)
            else:
                cluster_col = clusters.reshape(-1, 1)
        else:
            cluster = np.arange(len(self.embedding_list))
            cluster_col = cluster.reshape(-1, 1)
        return embeddings_array, cluster_col, scaler

    def find_top_similar(self, embeddings_array, cluster_col, scaler, current_inspiration_embedding=None):
        if current_inspiration_embedding is not None:
            current_inspiration_embedding_np = current_inspiration_embedding.detach().cpu().numpy()
            new_data_standardized = scaler.transform(current_inspiration_embedding_np.reshape(1, -1))
            distances = cdist(
                new_data_standardized, embeddings_array, 'euclidean')
            sorted_indices = np.argsort(distances[0])
        else:
            sorted_indices = np.arange(len(embeddings_array))

        top_indices = []
        visited_clusters = []

        for index in sorted_indices:
            cluster = cluster_col[index, 0]

            if cluster != -1 and cluster not in visited_clusters:
                top_indices.append(index)
                visited_clusters.append(cluster)
            elif cluster == -1:
                top_indices.append(index)

        return top_indices

    def select_inspirations(self, current_inspiration_list=None):
        if len(self.embedding_list) != 0:
            embeddings_array, cluster_col, scaler = self.analyze_cluster()
            if current_inspiration_list is not None:
                total_inspiration_indices = []
                for each_inspiration in current_inspiration_list:
                    simplified_inspiration = {
                        "Name": each_inspiration["Name"],
                        "Definition": each_inspiration["Definition"]
                    }
                    current_inspiration_str = json.dumps(simplified_inspiration)
                    current_inspiration_embedding = self._get_sentence_embedding(current_inspiration_str)
                    inspiration_indices = self.find_top_similar(embeddings_array, cluster_col, scaler, current_inspiration_embedding=current_inspiration_embedding)
                    total_inspiration_indices.extend(inspiration_indices)
            else:
                total_inspiration_indices = self.find_top_similar(embeddings_array, cluster_col, scaler)
            total_inspiration_indices = list(set(total_inspiration_indices))
            inspiration_to_return = [self.idea_pool[each_index] for each_index in total_inspiration_indices]
            rand_idx = np.random.permutation(len(inspiration_to_return))
            rand_idx = rand_idx[:self.paras.num_idea_to_return]
            return [inspiration_to_return[each_index] for each_index in rand_idx]
        else:
            return []

    def _prompt_parse_add(self, prompts_content):
        parsed_prompt, response_content, _ = self._prompt_till_valid(prompts_content)
        for every_new_inspiration in parsed_prompt:
            new_inspiration = {
                "Name": every_new_inspiration["Name"],
                "Definition": every_new_inspiration["Definition"],
                "Example": every_new_inspiration["Example"]
            }
            new_inspiration_str = json.dumps(new_inspiration)
            inspiration_str_embedding = self._get_sentence_embedding(new_inspiration_str)
            self.embedding_list.append(inspiration_str_embedding)
            self.idea_pool.append(new_inspiration)

        if len(self.embedding_list) > self.paras.pool_size:
            self.embedding_list = self.embedding_list[-self.paras.pool_size:]
            self.idea_pool = self.idea_pool[-self.paras.pool_size:]

    def summarize_offspring(self, parents, offspring):
        prompts_content = CoEvoPrompt.get_offspring_summarizer_prompt(
            self.task_info_inst, self.paras, parents, offspring, self.idea_pool)
        self._prompt_parse_add(prompts_content)

    def summarize(self, population):
        for each_indiv in population:
            self.summarize_indiv(each_indiv)

    def summarize_indiv(self, indiv):
        prompts_content = CoEvoPrompt.get_single_summarizer_prompt(
            self.task_info_inst, self.paras, indiv, self.idea_pool
        )
        self._prompt_parse_add(prompts_content)

    def _get_sentence_embedding(self, sentence):
        with torch.no_grad():
            t_input = self.tokenizer_inst(sentence, return_tensors="pt")
            t_input_dict = {key: tensor.to(self.device) for key, tensor in t_input.items()}
            with torch.no_grad():
                outputs = self.embedding_inst(**t_input_dict, output_hidden_states=True)
            sentence_embedding = outputs.hidden_states[-1][0]
            sentence_embedding = sentence_embedding.mean(dim=0)
        return sentence_embedding

    def _prompt_till_valid(self, prompt_content, **kwargs):
        n_retry = 0
        parse_success = False
        while not parse_success:
            response = self.paras.llm_summarizer_inst.get_response(prompt_content)
            parsed_response, parse_success = self._parse_response(response, **kwargs)

            if parse_success:
                return parsed_response, response, parse_success
            else:
                n_retry += 1

            if n_retry > 3:
                return None, response, False

    def _parse_response(self, response_str, **kwargs):
        try:
            idea_heading = r"(?:new|New|NEW)\s*(?:idea|Idea|IDEA)[sS]?\s*:?"

            # Parse Inspirations
            inspirations_pattern = re.compile(r"##\s*" + idea_heading + r"\s*(.*?)##\s*", re.DOTALL)
            match = inspirations_pattern.search(response_str)
            parsed_inspirations = []
            if match:
                inspirations_text = match.group(1).strip()
                inspiration_pattern = re.compile(
                    r"(?:reasoning|Reasoning|REASONING|reason|Reason|REASON)[sS]?\s*:?\s*(.*?)\s*\n.*?"
                    r"(?:name|Name|NAME)\s*:?\s*(.*?)\s*\n.*?"
                    r"(?:definition|Definition|DEFINITION)\s*:?\s*(.*?)\s*\n.*?"
                    r"(?:example|Example|EXAMPLE)[sS]?\s*:?\s*(.*?)\s*-*\s*(?=idea|Idea|IDEA|$)", re.DOTALL
                )
                inspirations = inspiration_pattern.findall(inspirations_text)
                for inspiration in inspirations:
                    parsed_inspirations.append({
                        "Reasoning": inspiration[0].strip(),
                        "Name": inspiration[1].strip(),
                        "Definition": inspiration[2].strip(),
                        "Example": inspiration[3].strip()
                    })
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None, False
        return parsed_inspirations, True