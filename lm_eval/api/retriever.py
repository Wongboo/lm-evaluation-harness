import time

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Retriever:
    def __init__(self,
                 embedding_model="/mnt/share/models/huggingface/bge-m3/",
                 embedding_device=0,
                 embedding_dtype='fp16',
                 embedding_pooling_method='cls',
                 index_addr="/data/ref/wiki_en+zh/index_m3_compress.faiss",
                 ref_dir="/data/ref/wiki_en+zh/arrow",
                 num_refs=3,
        ):
        if embedding_dtype == "bf16":
            dtype = torch.bfloat16
        elif embedding_dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.query_max_length = 512
        self.device = embedding_device
        self.pooling_method = embedding_pooling_method

        self.query_encoder = AutoModel.from_pretrained(embedding_model, torch_dtype=dtype).to(
            embedding_device)
        self.query_encoder.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = 'left'

        t0 = time.time()
        index = faiss.read_index(index_addr)
        # if getattr(config, 'gpu_index', False):
        #     if faiss.get_num_gpus() > 1:
        #         co = faiss.GpuMultipleClonerOptions()
        #         co.shard = True
        #         index = faiss.index_cpu_to_all_gpus(index, co=co)
        #     else:
        #         index = faiss.index_cpu_to_gpu(index, 0)
        print(f"Loaded vector index in time {time.time() - t0} sec")
        self.index = index
        self.ref_database = load_from_disk(ref_dir)
        self.num_refs = num_refs

    def encode(self, inputs):
        with torch.no_grad():
            if self.pooling_method == 'pooler_output':
                embeddings = self.query_encoder(**inputs).pooler_output
            elif self.pooling_method == "cls":
                embeddings = self.query_encoder(**inputs).last_hidden_state[:, 0]
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def search(self, query, hits):
        # TODO: mix retrieve strategy
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True,
                                max_length=self.query_max_length)
        inputs = inputs.to(self.device)
        embeddings = self.encode(inputs).cpu().numpy().astype(np.float32, order="C")
        faiss_results = self.index.search(embeddings, k=hits)
        return faiss_results


class Dummy:
    pass
