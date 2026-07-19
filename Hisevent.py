"""
HISEvent: Hierarchical and Incremental Structural Entropy Minimization
for Unsupervised Social Event Detection (Cao et al., AAAI 2024)

저자 원본 코드와 알고리즘 완전 동일.
데이터 매핑만 YouTube 댓글 스키마에 맞게 변경.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[v2 추가 최적화 요약] — 이전 버전 대비 추가 개선

  F) cross 캐시 증분 업데이트 (★★ 핵심 — 최대 효과)
     _calc_2d_entry 내부 _fast_cut(set(merged)) 완전 제거
     병합 시 cross[(vm1,vx)] 를 O(|comm_vm2| × deg) 증분으로 유지
     → update_division_MinSE 전체가 O(E) 이내로 수렴

  G) update_division_MinSE 내 cut 재계산 제거
     병합 후 cut(vm1) = cut(vm1_old) + cut(vm2) - 2×cross(vm1,vm2)
     _fast_cut(set(…)) 호출 0회

  H) struc_data / struc_data_2d 를 array-of-struct → struct-of-array
     numpy array 기반 배치 연산으로 vmnodeSE / vmSE 계산 벡터화

  I) node_to_comm 역매핑 증분 유지
     매 병합마다 전체 dict 재구축 제거 → O(|merged_comm|) 갱신

  J) heapq 중복 push 억제
     cross 변화가 없는 쌍은 재push 안 함

이전 버전(v1) 최적화:
  A) SE._deg / SE._adj 캐시
  B) heapq lazy-deletion
  C) hier_2D_SE_mini ProcessPoolExecutor
  D) get_graph_edges numpy 벡터화
  E) extract_named_entities_batch 멀티스레드

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[v3~v5 추가] 이벤트 Theme(Topic) 분류 — 임베딩 기반 Zero-shot 분류

  기존 BERTopic 키워드(단어 나열)는 "원전/전쟁/선거/성소수자" 같은
  상위 개념(카테고리) 분류에는 적합하지 않아 성능이 낮았음.

  → LLM 없이, 이벤트(클러스터)의 video_title(영상 제목)을 댓글 수로
    가중 평균한 임베딩과, 주제 앵커 문장 임베딩 간 코사인 유사도로
    이벤트 Theme을 분류.
    (댓글 텍스트 자체는 조롱/감정 표현이 많아 주제 설명력이 낮으므로
     뉴스 헤드라인 성격의 video_title을 분류 입력으로 사용)

  [v5] 임베딩 모델 통일
    기존 distiluse-base-multilingual-cased-v1(다국어 범용)은 짧은 한국어
    문장 간 STS(의미 유사도) 변별력이 약해, 이벤트 감지 + Theme 분류
    전체에서 jhgan/ko-sroberta-multitask(한국어 STS 특화)로 통일 교체.
    SBERT_embed() 하나만 유지하고, theme_embed()는 이를 감싸는 wrapper로
    단순화 (별도 모델 로딩 없음 → 메모리/로딩 비용 절감).

  - 결정적(deterministic) 결과 → 논문 재현성 확보
  - 외부 LLM 서버(Ollama) 불필요
  - 유사도 임계값(THEME_SIM_THRESHOLD) 미만이면 "기타"로 보수적 처리
  - THEME_ANCHORS 딕셔너리만 수정하면 카테고리 확장/조정 가능
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import re
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import heapq
import numpy as np
import pandas as pd
import networkx as nx
from itertools import chain
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from sklearn.metrics.cluster import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
)
from bertopic import BERTopic
from datetime import datetime
import random
from collections import defaultdict
from kiwipiepy import Kiwi

import faiss

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

KIWI = Kiwi()


# ══════════════════════════════════════════════════════════════
# 0. 유틸리티
# ══════════════════════════════════════════════════════════════

def fast_knn_edges(embeddings, k):
    x = np.array(embeddings).astype('float32')
    faiss.normalize_L2(x)
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    sim, idx = index.search(x, k + 1)
    edges = set()
    for i in range(len(x)):
        for j in idx[i][1:]:
            if i < j:
                edges.add((i + 1, j + 1))
    return list(edges)


def resolve_csv_path(date_str: str, base_dir: str = "comments_data") -> str:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError("날짜 형식은 YYYY-MM-DD 여야 합니다.")
    filename = f"comments_{date_str}.csv"
    full_path = os.path.join(base_dir, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {full_path}")
    return full_path


SBERT_MODEL_EN = None
SBERT_MODEL_MULTI = None


def SBERT_embed(s_list, language='Korean'):
    """
    [v5] 전체 파이프라인(이벤트 감지 + Theme 분류)에서 임베딩 모델 통일.
    Korean: distiluse-base-multilingual-cased-v1(다국어 범용) →
            jhgan/ko-sroberta-multitask(한국어 STS 특화)로 교체.
            한국어 문장 간 의미 유사도 변별력이 더 높아 이벤트 감지(KNN 그래프,
            stable point 탐색)와 Theme 분류 양쪽 품질에 도움이 됨.
    """
    global SBERT_MODEL_EN, SBERT_MODEL_MULTI
    if language == 'English':
        if SBERT_MODEL_EN is None:
            SBERT_MODEL_EN = SentenceTransformer('all-MiniLM-L6-v2',device='cpu')
        model = SBERT_MODEL_EN
    else:
        if SBERT_MODEL_MULTI is None:
            SBERT_MODEL_MULTI = SentenceTransformer(
                'jhgan/ko-sroberta-multitask',
                device='cpu'
            )
        model = SBERT_MODEL_MULTI
    embeddings = model.encode(
        s_list,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    return embeddings.cpu()


def evaluate(labels_true, labels_pred):
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    return nmi, ami, ari


def decode(division):
    if type(division) is dict:
        prediction_dict = {
            m: event for event, messages in division.items() for m in messages
        }
    elif type(division) is list:
        prediction_dict = {
            m: event for event, messages in enumerate(division) for m in messages
        }
    prediction_dict_sorted = dict(sorted(prediction_dict.items()))
    return list(prediction_dict_sorted.values())


# ══════════════════════════════════════════════════════════════
# 1. SE 클래스
#
# [v1 최적화 A] _deg / _adj 캐시 (networkx 반복 호출 제거)
# [v1 최적화 B] heapq lazy-deletion
#
# [v2 최적화 F] cross 캐시 증분 업데이트 ← 핵심 추가
#   self._cross[(v1,v2)]: 커뮤니티 v1↔v2 사이 엣지 가중치 합
#   - 초기화: update_struc_data_2d 와 동일한 단일 엣지 순회로 구축
#   - 병합 vm1∪vm2 후:
#       (a) vm2 관련 cross 엔트리 제거
#       (b) 모든 vx에 대해
#           cross[(vm1,vx)] += cross[(vm2,vx)]  (vm2 기여 흡수)
#       (c) node_to_comm 역매핑을 vm2 노드 → vm1 으로 갱신 (O(|vm2|))
#
# [v2 최적화 G] 병합 후 cut 재계산 제거
#   cut(vm1_new) = cut(vm1) + cut(vm2) - 2 × cross[(vm1,vm2)]
#   → _fast_cut(set(…)) 호출 0회
#
# [v2 최적화 H] _calc_2d_entry 에서도 cross 캐시 사용
#   gm = cut[v1] + cut[v2] - 2 × cross[(v1,v2)]
#   _fast_cut 완전 제거
#
# [v2 최적화 I] node_to_comm 증분 유지
#   병합마다 O(|merged_comm|) 갱신, 전체 재구축 없음
# ══════════════════════════════════════════════════════════════

class SE:
    def __init__(self, graph: nx.Graph):
        self.graph = graph.copy()
        # ── [최적화 A] degree / adjacency 캐시
        self._deg = {
            n: graph.degree(n, weight='weight') for n in graph.nodes
        }
        self._adj = {
            u: {v: data['weight'] for v, data in graph[u].items()}
            for u in graph.nodes
        }
        self.vol = sum(self._deg.values())
        self.division = {}
        self.struc_data = {}
        self.struc_data_2d = {}

        # ── [v2 최적화 F] cross 캐시 & node_to_comm 역매핑
        # division 초기화 후 _build_cross_cache() 로 구축
        self._cross: dict = {}           # (min_key, max_key) → float
        self._node_to_comm: dict = {}    # node → comm_key

    # ────────────────────────────────────────────
    # 캐시 기반 기본 연산
    # ────────────────────────────────────────────
    def _fast_cut(self, comm_set: set) -> float:
        return sum(
            w
            for u in comm_set
            for v, w in self._adj.get(u, {}).items()
            if v not in comm_set
        )

    def _fast_volume(self, comm) -> float:
        return sum(self._deg.get(n, 0) for n in comm)

    def get_cut(self, comm):
        return self._fast_cut(set(comm))

    def get_volume(self, comm):
        return self._fast_volume(comm)

    def get_vol(self):
        return sum(self._deg.values())

    # ────────────────────────────────────────────
    # [v2 최적화 F] cross 캐시 구축 (division 초기화 직후 1회)
    # ────────────────────────────────────────────
    def _build_cross_cache(self):
        """
        _adj를 1회 순회 → 커뮤니티 간 교차 가중치 cross[(ck1,ck2)] 구축.
        node_to_comm 도 동시에 구축.
        양방향(_adj에 u→v, v→u 모두 존재) 이므로 ÷2.
        """
        self._node_to_comm = {
            node: vn
            for vn, comm in self.division.items()
            for node in comm
        }
        cross = defaultdict(float)
        for u, nbrs in self._adj.items():
            cu = self._node_to_comm.get(u)
            if cu is None:
                continue
            for v, w in nbrs.items():
                cv = self._node_to_comm.get(v)
                if cv is None or cv == cu:
                    continue
                k = (cu, cv) if cu < cv else (cv, cu)
                cross[k] += w
        for k in cross:
            cross[k] *= 0.5
        self._cross = dict(cross)

    def _get_cross(self, v1, v2) -> float:
        k = (v1, v2) if v1 < v2 else (v2, v1)
        return self._cross.get(k, 0.0)

    # ────────────────────────────────────────────
    # 1D SE
    # ────────────────────────────────────────────
    def update_1dSE(self, original_1dSE, new_edges):
        affected_nodes = set()
        for edge in new_edges:
            affected_nodes.update([edge[0], edge[1]])

        original_vol = self.vol
        original_degree_dict = {
            n: self._deg.get(n, 0) for n in affected_nodes
        }

        self.graph.add_weighted_edges_from(new_edges)
        for u, v, w in new_edges:
            self._deg[u] = self._deg.get(u, 0) + w
            self._deg[v] = self._deg.get(v, 0) + w
            if u not in self._adj:
                self._adj[u] = {}
            if v not in self._adj:
                self._adj[v] = {}
            self._adj[u][v] = self._adj[u].get(v, 0) + w
            self._adj[v][u] = self._adj[v].get(u, 0) + w
        self.vol = sum(self._deg.values())
        updated_vol = self.vol

        if original_vol <= 0 or updated_vol <= 0:
            return original_1dSE

        updated_1dSE = (original_vol / updated_vol) * (
            original_1dSE - math.log2(original_vol / updated_vol)
        )
        for node in affected_nodes:
            d_original = original_degree_dict[node]
            d_updated = self._deg.get(node, 0)
            if d_original != d_updated:
                if d_original != 0:
                    updated_1dSE += (d_original / updated_vol) * math.log2(
                        d_original / updated_vol
                    )
                updated_1dSE -= (d_updated / updated_vol) * math.log2(
                    d_updated / updated_vol
                )
        return updated_1dSE

    def calc_1dSE(self):
        SE_val = 0
        for n in self.graph.nodes:
            d = self._deg.get(n, 0)
            if d > 0:
                SE_val += -(d / self.vol) * math.log2(d / self.vol)
        return SE_val

    # ────────────────────────────────────────────
    # struc_data (1D 파티션 통계)
    # ────────────────────────────────────────────
    def update_struc_data(self):
        self.struc_data = {}
        for vname in self.division:
            comm = self.division[vname]
            volume = self._fast_volume(comm)
            cut = self._fast_cut(set(comm))
            vSE = (
                0 if volume == 0
                else -(cut / self.vol) * math.log2(volume / self.vol)
            )
            vnodeSE = 0
            for node in comm:
                d = self._deg.get(node, 0)
                if d != 0:
                    vnodeSE -= (d / self.vol) * math.log2(d / volume)
            self.struc_data[vname] = [volume, cut, vSE, vnodeSE]

    # ────────────────────────────────────────────
    # [v2 최적화 H] _calc_2d_entry: cross 캐시 사용 → _fast_cut 제거
    # ────────────────────────────────────────────
    def _calc_2d_entry(self, v1, v2):
        """
        [v2] cross 캐시로 gm 계산 → O(1), _fast_cut 호출 없음.
        gm = cut(v1) + cut(v2) - 2 × cross(v1,v2)
        """
        gm = (
            self.struc_data[v1][1]
            + self.struc_data[v2][1]
            - 2.0 * self._get_cross(v1, v2)
        )
        vm = self.struc_data[v1][0] + self.struc_data[v2][0]

        if (
            self.struc_data[v1][0] <= 0
            or self.struc_data[v2][0] <= 0
            or vm <= 0
        ):
            vmSE = self.struc_data[v1][2] + self.struc_data[v2][2]
            vmnodeSE = self.struc_data[v1][3] + self.struc_data[v2][3]
        else:
            vmSE = -(gm / self.vol) * math.log2(vm / self.vol)
            vmnodeSE = (
                self.struc_data[v1][3]
                - (self.struc_data[v1][0] / self.vol)
                * math.log2(self.struc_data[v1][0] / vm)
                + self.struc_data[v2][3]
                - (self.struc_data[v2][0] / self.vol)
                * math.log2(self.struc_data[v2][0] / vm)
            )
        return [vm, gm, vmSE, vmnodeSE]

    # ────────────────────────────────────────────
    # struc_data_2d (cross 캐시 기반, O(k²) × O(1))
    # ────────────────────────────────────────────
    def update_struc_data_2d(self):
        """
        cross 캐시를 이용해 O(k²) × O(1) 로 전체 쌍 계산.
        _build_cross_cache() 가 먼저 호출되어 있어야 함.
        """
        self.struc_data_2d = {}
        comm_keys = list(self.division.keys())
        for i in range(len(comm_keys)):
            for j in range(i + 1, len(comm_keys)):
                v1, v2 = comm_keys[i], comm_keys[j]
                k = (v1, v2) if v1 < v2 else (v2, v1)
                self.struc_data_2d[k] = self._calc_2d_entry(v1, v2)

    # ────────────────────────────────────────────
    # division 초기화
    # ────────────────────────────────────────────
    def init_division(self):
        self.division = {}
        for node in self.graph.nodes:
            self.division[node] = [node]
            self.graph.nodes[node]['comm'] = node

    def add_isolates(self):
        all_nodes = sorted(list(chain(*list(self.division.values()))))
        edge_nodes = sorted(list(self.graph.nodes))
        if all_nodes != edge_nodes:
            for node in set(all_nodes) - set(edge_nodes):
                self.graph.add_node(node)
                self._deg[node] = 0
                self._adj[node] = {}

    # ────────────────────────────────────────────
    # delta 계산
    # ────────────────────────────────────────────
    def _mg_delta(self, v1, v2) -> float:
        k = (v1, v2) if v1 < v2 else (v2, v1)
        if k not in self.struc_data_2d:
            return float('inf')
        vm, gm, vmSE, vmnodeSE = self.struc_data_2d[k]
        return (
            vmSE + vmnodeSE
            - self.struc_data[v1][2] - self.struc_data[v1][3]
            - self.struc_data[v2][2] - self.struc_data[v2][3]
        )

    # ────────────────────────────────────────────
    # [v2 최적화 F+G] update_division_MinSE
    #   - _fast_cut 호출 0회
    #   - cross 캐시 증분 업데이트
    #   - node_to_comm 증분 유지
    # ────────────────────────────────────────────
    def update_division_MinSE(self):
        """
        [v2 핵심 최적화]

        병합 vm1 ∪ vm2 시:
          1. cut(vm1_new) = cut(vm1) + cut(vm2) - 2×cross[(vm1,vm2)]
             → _fast_cut 호출 없음  [최적화 G]

          2. cross 캐시 증분 업데이트  [최적화 F]
             for each vx ≠ vm1, vm2:
               cross[(vm1,vx)] += cross.get((vm2,vx), 0)
             vm2 관련 cross 엔트리 일괄 삭제

          3. node_to_comm 증분 유지  [최적화 I]
             vm2 노드 → vm1 으로 dict 갱신, O(|vm2|)

          4. struc_data_2d도 cross 캐시 사용 → O(1)/쌍  [최적화 H]

          5. heapq lazy-deletion  [v1 최적화 B]
        """
        # ── 초기 heap: 음수 delta 만 적재
        heap = []
        comm_keys = list(self.division.keys())
        for i in range(len(comm_keys)):
            for j in range(i + 1, len(comm_keys)):
                v1, v2 = comm_keys[i], comm_keys[j]
                d = self._mg_delta(v1, v2)
                if d < 0:
                    heapq.heappush(heap, (d, v1, v2))

        while heap:
            delta, vm1, vm2 = heapq.heappop(heap)

            # lazy deletion
            if vm1 not in self.division or vm2 not in self.division:
                continue

            cur = self._mg_delta(vm1, vm2)
            if cur >= 0:
                continue
            if abs(cur - delta) > 1e-12:
                heapq.heappush(heap, (cur, vm1, vm2))
                continue

            # ── [최적화 G] cut 재계산 없이 delta 공식으로 직접 계산
            cross_vm1_vm2 = self._get_cross(vm1, vm2)
            new_cut = (
                self.struc_data[vm1][1]
                + self.struc_data[vm2][1]
                - 2.0 * cross_vm1_vm2
            )
            new_volume = self.struc_data[vm1][0] + self.struc_data[vm2][0]

            if new_volume <= 0:
                new_vmSE = 0.0
            else:
                new_vmSE = -(new_cut / self.vol) * math.log2(new_volume / self.vol)

            sd1, sd2 = self.struc_data[vm1], self.struc_data[vm2]
            if sd1[0] <= 0 or sd2[0] <= 0 or new_volume <= 0:
                new_vmnodeSE = sd1[3] + sd2[3]
            else:
                new_vmnodeSE = (
                    sd1[3]
                    - (sd1[0] / self.vol) * math.log2(sd1[0] / new_volume)
                    + sd2[3]
                    - (sd2[0] / self.vol) * math.log2(sd2[0] / new_volume)
                )

            # ── 병합 실행
            comm_vm2 = self.division[vm2]
            for node in comm_vm2:
                self.graph.nodes[node]['comm'] = vm1
                self._node_to_comm[node] = vm1           # [최적화 I]
            self.division[vm1] = self.division[vm1] + comm_vm2
            del self.division[vm2]

            self.struc_data[vm1] = [new_volume, new_cut, new_vmSE, new_vmnodeSE]
            del self.struc_data[vm2]

            # ── [최적화 F] cross 캐시 증분 업데이트
            # vm2 관련 키 수집
            k_vm1_vm2 = (vm1, vm2) if vm1 < vm2 else (vm2, vm1)
            keys_to_delete = [
                k for k in self._cross
                if k[0] == vm2 or k[1] == vm2
            ]
            for k in keys_to_delete:
                # vx = 상대방 커뮤니티
                vx = k[1] if k[0] == vm2 else k[0]
                if vx == vm1:
                    del self._cross[k]
                    continue
                w = self._cross.pop(k, 0.0)
                # vm1↔vx 에 vm2↔vx 기여 흡수
                km1x = (vm1, vx) if vm1 < vx else (vx, vm1)
                self._cross[km1x] = self._cross.get(km1x, 0.0) + w

            # ── vm2 관련 struc_data_2d 제거
            keys_2d_del = [
                kk for kk in self.struc_data_2d
                if kk[0] == vm2 or kk[1] == vm2
            ]
            for kk in keys_2d_del:
                del self.struc_data_2d[kk]

            # ── vm1 인접 쌍 struc_data_2d 재계산 (cross 캐시 O(1))
            for vx in list(self.division.keys()):
                if vx == vm1:
                    continue
                k = (vm1, vx) if vm1 < vx else (vx, vm1)
                self.struc_data_2d[k] = self._calc_2d_entry(vm1, vx)
                nd = self._mg_delta(vm1, vx)
                if nd < 0:
                    heapq.heappush(heap, (nd, vm1, vx))


# ══════════════════════════════════════════════════════════════
# 2. 그래프 구성 & SE 최소화
# ══════════════════════════════════════════════════════════════

def search_stable_points(embeddings, max_num_neighbors=80):
    x = np.array(embeddings).astype('float32')
    faiss.normalize_L2(x)
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    sim_all, idx_all = index.search(x, max_num_neighbors + 1)

    knn_cache = {}
    all_1dSEs = []
    seg = None

    for i in range(max_num_neighbors):
        knn_edges = []
        for s in range(len(x)):
            d = idx_all[s, i + 1]
            score = float(sim_all[s, i + 1])
            if score > 0:
                knn_edges.append((s + 1, d + 1, score))
        knn_cache[i + 1] = knn_edges

        if i == 0:
            g = nx.Graph()
            g.add_weighted_edges_from(knn_edges)
            seg = SE(g)
            all_1dSEs.append(seg.calc_1dSE())
        else:
            all_1dSEs.append(seg.update_1dSE(all_1dSEs[-1], knn_edges))

    stable_indices = [
        i for i in range(1, len(all_1dSEs) - 1)
        if all_1dSEs[i] < all_1dSEs[i - 1] and all_1dSEs[i] < all_1dSEs[i + 1]
    ]

    if not stable_indices:
        return 0, 0, knn_cache

    stable_SEs = [all_1dSEs[i] for i in stable_indices]
    best = stable_indices[int(np.argmin(stable_SEs))]
    return stable_indices[0] + 1, best + 1, knn_cache


def get_graph_edges(attributes):
    """
    [최적화 D] numpy 벡터화
    """
    attr_nodes_dict = {}
    attr_weight_dict = {}
    attr_type_count = defaultdict(int)

    for i, attr_list in enumerate(attributes):
        for attr, weight in attr_list:
            attr_type = attr.split("::", 1)[0]
            attr_type_count[attr_type] += 1
            if attr not in attr_nodes_dict:
                attr_nodes_dict[attr] = [i + 1]
                attr_weight_dict[attr] = weight
            else:
                attr_nodes_dict[attr].append(i + 1)

    all_us, all_vs, all_ws = [], [], []
    VIDEO_K, NE_K = 10, 10
    LIMITED_ATTRS = {"video": VIDEO_K, "ne": NE_K}
    edge_stats = defaultdict(int)

    for attr, nodes in attr_nodes_dict.items():
        weight = attr_weight_dict[attr]
        attr_type = attr.split("::", 1)[0]

        if attr_type in LIMITED_ATTRS:
            K = LIMITED_ATTRS[attr_type]
            n_nodes = len(nodes)
            if n_nodes < 2:
                continue
            for u in nodes:
                neighbors = random.sample(nodes, min(K + 1, n_nodes))
                for v in neighbors:
                    if u == v:
                        continue
                    mn, mx = (u, v) if u < v else (v, u)
                    all_us.append(mn)
                    all_vs.append(mx)
                    all_ws.append(weight)
                    edge_stats[attr_type] += 1
        else:
            arr = np.array(nodes)
            if len(arr) < 2:
                continue
            idx_i, idx_j = np.triu_indices(len(arr), k=1)
            us = arr[idx_i]
            vs = arr[idx_j]
            mn = np.minimum(us, vs)
            mx = np.maximum(us, vs)
            edge_stats[attr_type] += len(mn)
            all_us.append(mn)
            all_vs.append(mx)
            all_ws.append(np.full(len(mn), weight))

    print("\n========== ATTRIBUTE STATS ==========")
    for t in ['author', 'video', 'parent', 'ne']:
        print(f"{t:<10} attributes : {attr_type_count[t]:,}")
    print("\n========== EDGE STATS ==========")
    for t in ['author', 'video', 'parent', 'ne']:
        print(f"{t:<10} edges      : {edge_stats[t]:,}")
    print(f"\nTotal attribute edges : {sum(edge_stats.values()):,}")

    if not all_us:
        return []

    us_cat = np.concatenate([np.atleast_1d(x) for x in all_us]).astype(np.int64)
    vs_cat = np.concatenate([np.atleast_1d(x) for x in all_vs]).astype(np.int64)
    ws_cat = np.concatenate([np.atleast_1d(x) for x in all_ws]).astype(np.float64)

    MAX_NODE = int(max(us_cat.max(), vs_cat.max())) + 1
    keys = us_cat * MAX_NODE + vs_cat
    sorted_keys, inverse = np.unique(keys, return_inverse=True)
    summed_w = np.zeros(len(sorted_keys), dtype=np.float64)
    np.add.at(summed_w, inverse, ws_cat)

    u_out = (sorted_keys // MAX_NODE).tolist()
    v_out = (sorted_keys % MAX_NODE).tolist()
    w_out = summed_w.tolist()
    return list(zip(u_out, v_out, w_out))


def get_knn_edges(embeddings, default_num_neighbors):
    return fast_knn_edges(embeddings, default_num_neighbors)


def get_global_edges(attributes, embeddings, default_num_neighbors,
                     e_a=True, e_s=True):
    graph_edges, knn_edges = [], []
    if e_a:
        graph_edges = get_graph_edges(attributes)
    if e_s:
        knn_edges = get_knn_edges(embeddings, default_num_neighbors)
    return list(set(knn_edges + graph_edges))


# ══════════════════════════════════════════════════════════════
# 3. hier_2D_SE_mini 병렬화
#    [v1 최적화 C] ProcessPoolExecutor
#    [v2 추가] cross 캐시 초기화(_build_cross_cache) 포함
# ══════════════════════════════════════════════════════════════

def _process_subgraph(args):
    """
    ProcessPoolExecutor worker.
    [v2] division 설정 후 _build_cross_cache() 호출 → cross 캐시 활성화.
    """
    subgraph_edges, sub_clusters = args
    g = nx.Graph()
    g.add_weighted_edges_from(subgraph_edges)
    seg = SE(g)
    seg.division = {j: cluster for j, cluster in enumerate(sub_clusters)}
    seg.add_isolates()
    for k in seg.division:
        for node in seg.division[k]:
            seg.graph.nodes[node]['comm'] = k
    seg.update_struc_data()

    # [v2] cross 캐시 구축 (update_struc_data_2d 이전에 반드시 호출)
    seg._build_cross_cache()
    seg.update_struc_data_2d()
    seg.update_division_MinSE()
    return list(seg.division.values())


def hier_2D_SE_mini(weighted_global_edges, n_messages, n=100):
    """
    [v1 최적화 C] ProcessPoolExecutor로 서브그래프 병렬 처리.
    알고리즘 흐름은 저자 코드와 완전 동일.
    """
    import multiprocessing
    max_workers = max(1, int(multiprocessing.cpu_count() * 0.75))

    no_progress_limit = 3
    no_progress_count = 0
    prev_n_clusters = n_messages
    MAX_N = 400

    edge_index = defaultdict(list)
    for edge in weighted_global_edges:
        u, v = edge[0], edge[1]
        edge_index[u].append(edge)
        edge_index[v].append(edge)

    def _get_subgraphs_edges(clusters, graph_splits):
        result = []
        for split in graph_splits:
            sub_clusters = clusters[split[0]:split[1]]
            subgraph_nodes = set(chain(*sub_clusters))
            seen = set()
            subgraph_edges = []
            for node in subgraph_nodes:
                for edge in edge_index[node]:
                    u, v = edge[0], edge[1]
                    key = (min(u, v), max(u, v))
                    if key not in seen and u in subgraph_nodes and v in subgraph_nodes:
                        seen.add(key)
                        subgraph_edges.append(edge)
            result.append(subgraph_edges)
        return result

    ite = 1
    clusters = [[i + 1] for i in range(n_messages)]

    while True:
        print(f'\n=========Iteration {ite}=========')
        n_clusters = len(clusters)
        graph_splits = [
            (s, min(s + n, n_clusters)) for s in range(0, n_clusters, n)
        ]
        all_subgraphs_edges = _get_subgraphs_edges(clusters, graph_splits)
        last_clusters = clusters

        if len(graph_splits) == 1:
            results = [
                _process_subgraph((
                    all_subgraphs_edges[0],
                    last_clusters[graph_splits[0][0]:graph_splits[0][1]]
                ))
            ]
        else:
            args_list = [
                (
                    all_subgraphs_edges[i],
                    last_clusters[graph_splits[i][0]:graph_splits[i][1]]
                )
                for i in range(len(graph_splits))
            ]
            results = [None] * len(graph_splits)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(_process_subgraph, a): idx
                    for idx, a in enumerate(args_list)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()

        clusters = []
        for res in results:
            clusters.extend(res)

        if len(graph_splits) == 1:
            break

        cur_n_clusters = len(clusters)
        if cur_n_clusters < prev_n_clusters:
            no_progress_count = 0
        else:
            no_progress_count += 1
            print(f'  → 진전 없음 ({no_progress_count}/{no_progress_limit})')
            if no_progress_count >= no_progress_limit:
                print(f'  → {no_progress_limit}회 연속 진전 없음. 종료.')
                break
        prev_n_clusters = cur_n_clusters

        if clusters == last_clusters:
            old_n = n
            n = min(n * 2, MAX_N)
            if n > old_n:
                print(f'  → n 증가: {n}')
            else:
                print(f'  → n 최대값({MAX_N}) 유지')

        ite += 1

    return clusters


# ══════════════════════════════════════════════════════════════
# 4. Named Entity 추출
#    [v1 최적화 E] ThreadPoolExecutor 멀티스레드
# ══════════════════════════════════════════════════════════════

# 전역 Kiwi 인스턴스 재사용 (스레드별 신규 생성 X)
def _kiwi_analyze_chunk(chunk):
    TARGET_TAGS = {"NNP", "SL", "SH"}
    result = []
    for analyzed in KIWI.analyze(chunk):
        tokens = analyzed[0][0]
        entities = [
            t.form for t in tokens
            if t.tag in TARGET_TAGS and len(t.form) >= 2
        ]
        result.append(entities)
    return result


def extract_named_entities_batch(texts, freq_threshold=0.01):
    import multiprocessing
    n_workers = min(4, max(1, multiprocessing.cpu_count() // 3))
    print(f"[NER] Kiwi 멀티스레드 ({n_workers} workers) NNP/SL/SH 추출 중...")

    chunk_size = max(1, math.ceil(len(texts) / n_workers))
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    raw_result_chunks = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(_kiwi_analyze_chunk, chunk): idx
            for idx, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            raw_result_chunks[idx] = future.result()

    raw_result = []
    for chunk_res in raw_result_chunks:
        raw_result.extend(chunk_res)

    n_docs = len(texts)
    df_counter = defaultdict(int)
    for entities in raw_result:
        for entity in set(entities):
            df_counter[entity] += 1

    cutoff = freq_threshold * n_docs
    stopword_nes = {ne for ne, df in df_counter.items() if df >= cutoff}
    print(f"[NER] 고빈도 제거: {len(stopword_nes):,}개")

    result = [
        [e for e in entities if e not in stopword_nes]
        for entities in raw_result
    ]

    print("\nNER Sample")
    for i in range(min(10, len(result))):
        print(result[i])

    return result


# ══════════════════════════════════════════════════════════════
# 5. YouTube 댓글 속성 추출
# ══════════════════════════════════════════════════════════════

def build_youtube_attributes(messages, named_entities_list, video_keep_ratio=1.0):
    video_groups = defaultdict(list)
    for i, msg in enumerate(messages):
        vid = msg.get('video_id')
        if vid:
            video_groups[vid].append(i)

    attributes = [[] for _ in messages]

    for vid, idxs in video_groups.items():
        k = max(1, int(len(idxs) * video_keep_ratio))
        sampled = random.sample(idxs, k)
        for i in sampled:
            attributes[i].append(("video::" + str(vid), 1.0))

    for i, msg in enumerate(messages):
        if msg.get('author_name'):
            attributes[i].append(("author::" + str(msg["author_name"]), 1.0))
        pid = msg.get('parent_comment_id')
        if pid and str(pid).strip():
            attributes[i].append(("parent::" + str(pid), 1.0))
        for entity in named_entities_list[i]:
            attributes[i].append(("ne::" + str(entity), 1.0))

    return attributes


# ══════════════════════════════════════════════════════════════
# 6. 가중 엣지 구성
# ══════════════════════════════════════════════════════════════

def build_weighted_global_edges(attributes, semantic_knn_edges):
    ea_edges = get_graph_edges(attributes)

    edge_weight_map = {}
    for u, v, w in ea_edges:
        key = (u, v)
        edge_weight_map[key] = edge_weight_map.get(key, 0) + w

    for u, v, w in semantic_knn_edges:
        if w > 0:
            key = (min(u, v), max(u, v))
            edge_weight_map[key] = edge_weight_map.get(key, 0) + w

    return [(u, v, w) for (u, v), w in edge_weight_map.items()]


# ══════════════════════════════════════════════════════════════
# 7. 메인 실행
# ══════════════════════════════════════════════════════════════

CLUSTER_NODE_DIR = "cluster_node_20260716"


def _safe_filename(name: str, max_len: int = 80) -> str:
    """
    파일명으로 쓸 수 없는 문자를 제거/치환.
    """
    name = str(name).strip()
    name = re.sub(r'[\\/:*?"<>|\n\r\t]', '_', name)
    name = re.sub(r'\s+', ' ', name).strip()
    if not name:
        name = "unknown"
    return name[:max_len]


def save_cluster_comments_by_top_video(
    cluster_df: pd.DataFrame,
    cluster_id,
    date_str: str,
    theme: str = "기타",
    output_dir: str = CLUSTER_NODE_DIR,
):
    """
    이벤트로 감지된 클러스터(cluster_node)의 댓글 전체를,
    해당 클러스터에서 댓글 수가 가장 많은 video_title(top video)
    이름으로 CSV에 저장한다.

    추가:
      - top video가 속한 채널의 정치성향(channel_ideology) 컬럼 저장
      - Theme은 컬럼이 아닌 파일명에 포함
    """
    os.makedirs(output_dir, exist_ok=True)

    cluster_df = cluster_df.copy()

    if "video_title" in cluster_df.columns and cluster_df["video_title"].notna().any():

        # 가장 댓글이 많은 영상
        top_video = cluster_df["video_title"].value_counts().idxmax()

        # 그 영상의 channel_id
        top_channel_id = (
            cluster_df.loc[
                cluster_df["video_title"] == top_video,
                "channel_id"
            ]
            .mode()
            .iloc[0]
        )

        # 성향
        ideology = CHANNEL_INFO.get(
            top_channel_id,
            {"ideology": "기타"}
        )["ideology"]

    else:
        top_video = f"cluster_{cluster_id}"
        ideology = "기타"

    # 클러스터 전체에 같은 성향 기록
    cluster_df["channel_ideology"] = ideology

    safe_name = _safe_filename(top_video)
    safe_theme = _safe_filename(theme, max_len=30)
    cluster_size = len(cluster_df)

    filename = (
        f"{date_str}_{cluster_id}_{cluster_size}_"
        f"{ideology}_{safe_theme}_{safe_name}.csv"
    )
    filepath = os.path.join(output_dir, filename)

    cluster_df.to_csv(filepath, index=False, encoding="utf-8-sig")

    print(
        f"  → [저장] Cluster {cluster_id} "
        f"({cluster_size:,}개 댓글, {ideology}, Theme={theme}) → {filepath}"
    )

    return filepath


CHANNEL_INFO = {
    "UCLiK08tsWKL5bNoz1f8sHsw": {"name": "민중의소리 MediaVop",        "ideology": "진보"},
    "UCu1FzjrHosuKGvgIx8oBi8w": {"name": "[공식] 새날",               "ideology": "진보"},
    "UCNJM6dqu70Qr6VaseiW1Org": {"name": "이재명",              "ideology": "진보"},
    "UCNRDGWEilNZOIcVIq-FRKZQ": {"name": "빨간아재",               "ideology": "진보"},
    "UCMYhq9OyGI5UEz_NTAoHY7A": {"name": "[팟빵] 최욱의 매불쇼",                  "ideology": "진보"},
    "UCdp4_yTBhQmB8E339Lafzow": {"name": "조선일보",              "ideology": "보수"},
    "UCxuf3GXK290vcpFW0lxm0Uw": {"name": "이봉규TV",              "ideology": "보수"},
    "UCbgqDFODvh-q38ouzoKKkVA": {"name": "신인균의 국방TV",              "ideology": "보수"},
    "UC45hk7RSPPS_tezwn5rcL0A": {"name": "전옥현 안보정론TV",       "ideology": "보수"},
    "UC8p2BFLsMGSfBdpwNs4Zq1A": {"name": "뉴스데일리베스트",      "ideology": "보수"},
    "UC4Aa3OPkMenwTANpf0oWVRQ": {"name": "박성태의 뉴스쇼",        "ideology": "중도"},
    "UC1ifSsWUG241rRfK0ezYCgA": {"name": "KNN NEWS",             "ideology": "중도"},
    "UCV9gIwasozKlwz5U2Sa2q6g": {"name": "박재홍의 한판승부",       "ideology": "중도"},
    "UCHBvfByzuzamrZiL398jopQ": {"name": "뉴스1TV",               "ideology": "중도"},
    "UCRofX42ugIM1JKnCjfXm9LA": {"name": "채널A 김진의 돌직구쇼",   "ideology": "중도"},
}


# ══════════════════════════════════════════════════════════════
# 7-1. [v3 신규] 이벤트 Theme 분류 — 임베딩 기반 Zero-shot 분류
#
#   방법: 이벤트 감지에 이미 사용된 SBERT 임베딩을 재사용.
#     1) 카테고리별로 여러 개의 "앵커 문장"(자연어 설명문)을 정의
#     2) 앵커 문장들을 SBERT_embed()로 임베딩 (최초 1회만 계산 후 캐시)
#     3) 이벤트(클러스터) 소속 댓글들의 임베딩 centroid(평균)를 계산
#     4) centroid와 모든 앵커 임베딩 간 코사인 유사도를 구하고,
#        그 중 최댓값을 갖는 앵커가 속한 카테고리를 이벤트의 Theme으로 채택
#
#   LLM 미사용 → 결정적(deterministic), 외부 서버 불필요,
#   이미 계산된 임베딩을 재사용하므로 추가 비용이 거의 없음.
#
#   카테고리/앵커 문장은 필요에 따라 자유롭게 추가·수정 가능.
#   (한 카테고리에 여러 앵커 문장을 넣을수록 분류 강건성이 올라감)
# ══════════════════════════════════════════════════════════════

THEME_ANCHORS = {
    "원전 에너지": [
        "원자력 발전소와 원전 정책에 대한 논쟁",
        "탈원전과 에너지 정책 관련 이슈",
        "전력 수급과 원전 안전성 문제",
    ],
    "전쟁 안보": [
        "전쟁과 군사 충돌에 대한 뉴스",
        "북한의 군사 도발과 안보 위협",
        "국방과 군사 안보 관련 사건",
        "우크라이나 러시아 전쟁과 국제 분쟁",
    ],
    "성소수자 젠더": [
        "성소수자 인권과 차별금지법 논쟁",
        "동성혼과 젠더 이슈에 대한 사회적 갈등",
        "페미니즘과 젠더 갈등 관련 사건",
    ],
    "부동산 경제": [
        "부동산 가격과 주택 정책 논란",
        "경제 정책과 물가, 금리 이슈",
        "주식 시장과 경제 위기 관련 뉴스",
    ],
    "정치": [
        "대통령 선거와 총선 등 선거 이슈",
        "여야 정당 간의 정치적 갈등",
        "정치인의 발언과 정치적 사건",
        "대선 후보 유세와 선거 캠페인",
        "정권교체와 정치적 지지 호소",
        "정치인을 둘러싼 스캔들과 의혹 폭로",
        "정치인의 사생활 논란과 폭로성 인터뷰",
        "정치인 가족과 관련된 의혹 사건",
        "언론 보도와 가짜뉴스 논란",
        "방송사와 미디어의 편파성 논쟁",
        "검찰 수사와 기소를 둘러싼 논란",
        "법원 판결과 재판 관련 사건",
        "정치인에 대한 사법 처리 이슈",
    ],
    "노동 노사": [
        "노동조합 파업과 노사 갈등",
        "근로자 권리와 노동 정책 논란",
    ],
    "외교": [
        "한미 관계와 외교 정책 이슈",
        "한중, 한일 외교 갈등 사건",
        "국제 정상회담과 외교 협상",
    ],
    "복지 사회": [
        "복지 정책과 사회 안전망 논쟁",
        "저출산, 육아, 연금 등 사회 이슈",
    ],
    "방역 보건": [
        "코로나19 방역 정책과 백신 패스 논쟁",
        "감염병 대응과 의료 전문가 간의 논쟁",
        "사회적 거리두기와 방역 지침 논란",
    ],
    "환경": [
        "기후 변화와 환경 정책 이슈",
        "미세먼지와 환경 오염 문제",
    ],
    "교육": [
        "교육 정책과 입시 제도 논란",
        "학교와 학생 관련 사회적 이슈",
    ],
    "재난 사고": [
        "대형 참사와 재난 사고 뉴스",
        "안전사고와 재난 대응 논란",
    ],
    "기타": [
        "위 어느 주제에도 해당하지 않는 일반적인 사건",
    ],
}

# 최고 유사도가 이 값 미만이면 억지로 카테고리를 배정하지 않고 "기타"로 처리
THEME_SIM_THRESHOLD = 0.30

# 앵커 임베딩 캐시 (최초 1회만 계산)
_THEME_ANCHOR_CACHE = {}


def get_theme_anchor_embeddings(language: str = 'Korean'):
    """
    THEME_ANCHORS의 모든 앵커 문장을 Theme 분류 전용 모델(theme_embed)로 임베딩하고,
    (theme_name_per_anchor: list[str], anchor_embeddings: np.ndarray) 형태로 캐싱.
    """
    cache_key = "_default"
    if cache_key in _THEME_ANCHOR_CACHE:
        return _THEME_ANCHOR_CACHE[cache_key]

    anchor_sentences = []
    anchor_theme_names = []
    for theme, sentences in THEME_ANCHORS.items():
        for s in sentences:
            anchor_sentences.append(s)
            anchor_theme_names.append(theme)

    anchor_embeddings = theme_embed(anchor_sentences)
    # 코사인 유사도 계산을 내적으로 대체할 수 있도록 L2 정규화
    norms = np.linalg.norm(anchor_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    anchor_embeddings = anchor_embeddings / norms

    _THEME_ANCHOR_CACHE[cache_key] = (anchor_theme_names, anchor_embeddings)
    return anchor_theme_names, anchor_embeddings


# ── [v5] 임베딩 모델 통일: Theme 분류도 메인 SBERT_embed(ko-sroberta-multitask)를
#    그대로 재사용. 별도 모델 로딩 없음 → 메모리/로딩 비용 절감.
def theme_embed(s_list):
    """SBERT_embed()를 감싼 얇은 wrapper. numpy 배열로 반환."""
    return SBERT_embed(s_list, language='Korean').numpy()


def build_video_title_embedding_cache(video_titles, language: str = 'Korean'):
    """
    event_df에 등장하는 고유 video_title들을 SBERT로 임베딩(L2 정규화)하여
    {title: embedding} 딕셔너리로 반환. 클러스터별로 재계산하지 않고
    전체 이벤트에 대해 1회만 계산 후 재사용한다.
    """
    unique_titles = [t for t in pd.unique(pd.Series(video_titles).dropna()) if str(t).strip()]
    if not unique_titles:
        return {}

    title_embeddings = theme_embed(unique_titles)
    norms = np.linalg.norm(title_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    title_embeddings = title_embeddings / norms

    return {title: title_embeddings[i] for i, title in enumerate(unique_titles)}


def classify_event_theme_by_embedding(video_title_counts: dict,
                                       video_title_embed_cache: dict,
                                       anchor_theme_names,
                                       anchor_embeddings: np.ndarray,
                                       fallback_embeddings: np.ndarray = None,
                                       sim_threshold: float = THEME_SIM_THRESHOLD):
    """
    이벤트(클러스터)에 속한 video_title들의 "댓글 수 가중 평균 임베딩"을 계산해
    앵커 임베딩과의 코사인 유사도가 가장 높은 Theme을 반환.

    댓글 텍스트 자체(특히 정치 유튜브)는 조롱/감정 표현이 많아 주제 설명력이
    낮으므로, 뉴스 헤드라인 성격의 video_title을 분류 입력으로 사용한다.
    video_title 정보가 전혀 없는 경우에만 fallback_embeddings(댓글 centroid)를 사용.

    video_title_counts       : {title: comment_count} (해당 클러스터 내)
    video_title_embed_cache  : {title: L2정규화된 임베딩 벡터} (전체 event_df 기준 사전 계산)
    anchor_theme_names       : len == n_anchors
    anchor_embeddings        : (n_anchors, dim), L2 정규화됨
    fallback_embeddings      : video_title 정보가 없을 때 사용할 댓글 임베딩 배열
    sim_threshold            : 최고 유사도가 이 값 미만이면 "기타" 반환
    """
    weighted_sum = None
    weight_total = 0.0

    for title, count in video_title_counts.items():
        emb = video_title_embed_cache.get(title)
        if emb is None:
            continue
        w = float(count)
        if weighted_sum is None:
            weighted_sum = emb * w
        else:
            weighted_sum = weighted_sum + emb * w
        weight_total += w

    if weighted_sum is None or weight_total == 0.0:
        # video_title 정보가 없는 경우 → 댓글 centroid로 대체
        if fallback_embeddings is None or len(fallback_embeddings) == 0:
            return "기타", 0.0
        centroid = fallback_embeddings.mean(axis=0)
    else:
        centroid = weighted_sum / weight_total

    norm = np.linalg.norm(centroid)
    if norm == 0:
        return "기타", 0.0
    centroid = centroid / norm

    sims = anchor_embeddings @ centroid  # (n_anchors,)
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    if best_score < sim_threshold:
        return "기타", best_score
    return anchor_theme_names[best_idx], best_score


def run(
    csv_path: str,
    language: str = 'Korean',
    n: int = 100,
    max_num_neighbors: int = 80,
    output_path: str = 'hisevent_results.csv',
):
    df = pd.read_csv(csv_path)
    for col in ['author_name', 'video_id', 'parent_comment_id']:
        if col not in df.columns:
            df[col] = ''
        else:
            df[col] = df[col].fillna('')
    df['comment_text'] = df['comment_text'].fillna('')

    # [v3 신규] 임베딩 배열 인덱싱을 위한 원본 행 위치 기록
    # (이후 event_df = df[...] 로 서브셋을 만들어도 이 값으로
    #  embeddings_np에서 올바른 행을 다시 찾아올 수 있음)
    df['_row_idx'] = np.arange(len(df))

    messages = df.to_dict(orient='records')
    n_messages = len(messages)
    print(f'[Data] 로드된 댓글 수: {n_messages}')

    print('[Step 2] SBERT 임베딩...')
    embeddings = SBERT_embed(
        [m['comment_text'] for m in messages], language=language
    )
    # [v3 신규] 이벤트 Theme 분류를 위해 numpy 배열로도 보관
    embeddings_np = embeddings.numpy()

    print('[Step 3] Stable point 탐색 (증분 1D SE)...')
    first_stable_k, global_stable_k, knn_cache = search_stable_points(
        embeddings, max_num_neighbors=max_num_neighbors
    )
    default_num_neighbors = min(
        first_stable_k if first_stable_k > 0 else 1, 15
    )
    print(f'  → 사용할 k: {default_num_neighbors} (첫 번째 stable point)')

    print('[Step 4] Named Entity 추출 (Kiwi NER)...')
    named_entities_list = extract_named_entities_batch(
        [m['comment_text'] for m in messages]
    )

    print('[Step 5] 속성 기반 Ea 엣지 구성...')
    attributes = build_youtube_attributes(messages, named_entities_list)

    semantic_edges = [(u, v) for u, v, score in knn_cache[default_num_neighbors]]
    print(f"Semantic edges        : {len(semantic_edges):,}")

    print('[Step 6] 가중 글로벌 엣지 구성 (Ea ∪ Es)...')
    weighted_global_edges = build_weighted_global_edges(
        attributes, knn_cache[default_num_neighbors]
    )
    print(f'  → 전체 엣지 수: {len(weighted_global_edges)}')

    print(f'[Step 7] 계층적 2D SE 최소화 (n={n})...')
    clusters = hier_2D_SE_mini(weighted_global_edges, n_messages, n=n)

    labels = decode(clusters)
    df['event_id'] = labels

    cluster_counts = df['event_id'].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= 2000].index
    event_df = df[df['event_id'].isin(valid_clusters)].copy()

    # [v3 신규] 이벤트 Theme 분류용 앵커 임베딩 준비 (최초 1회 계산 후 캐시)
    print('[Step 7-1] 이벤트 Theme 분류용 앵커 임베딩 준비...')
    anchor_theme_names, anchor_embeddings = get_theme_anchor_embeddings(language=language)

    # [v3 신규] video_title 임베딩 캐시 (댓글 대신 영상 제목으로 분류하기 위함)
    # 정치 유튜브 댓글은 조롱/감정 표현이 많아 주제 설명력이 낮으므로,
    # 뉴스 헤드라인 성격의 video_title 기반 분류가 훨씬 안정적임.
    video_title_embed_cache = {}
    if "video_title" in event_df.columns:
        video_title_embed_cache = build_video_title_embedding_cache(
            event_df["video_title"].tolist(), language=language
        )

    print("\n==============================")
    print("EVENT TITLES")
    print("==============================")

    event_titles = {}
    event_themes = {}  # [v3 신규] cluster_id → theme 매핑
    event_theme_scores = {}  # [v3 신규] cluster_id → 유사도 점수 (참고용)

    for cluster_id in valid_clusters:
        cluster_df = event_df[event_df["event_id"] == cluster_id].copy()
        ideology_counts = (
            cluster_df["channel_id"]
            .map(lambda x: CHANNEL_INFO.get(x, {"ideology": "기타"})["ideology"])
            .fillna("기타")
            .value_counts()
        )
        ideology_text = ", ".join(f"{k}:{v}" for k, v in ideology_counts.items())
        cluster_comments = cluster_df["comment_text"].astype(str).tolist()

        # ── [v3] 이벤트 Theme 분류 (video_title 가중 임베딩 기반, LLM 미사용)
        # → 저장 호출 전에 먼저 계산해서 파일명에 사용
        if "video_title" in cluster_df.columns:
            title_counts = cluster_df["video_title"].value_counts().to_dict()
        else:
            title_counts = {}

        cluster_row_idx = cluster_df["_row_idx"].to_numpy()
        cluster_embs_fallback = embeddings_np[cluster_row_idx]  # video_title 없을 때만 사용

        theme, theme_score = classify_event_theme_by_embedding(
            video_title_counts=title_counts,
            video_title_embed_cache=video_title_embed_cache,
            anchor_theme_names=anchor_theme_names,
            anchor_embeddings=anchor_embeddings,
            fallback_embeddings=cluster_embs_fallback,
        )
        event_themes[cluster_id] = theme
        event_theme_scores[cluster_id] = theme_score
        print(f"Theme: {theme}  (유사도={theme_score:.4f})")

        # ── 이벤트로 감지된 클러스터의 댓글을 top video 이름 + theme으로 저장
        save_cluster_comments_by_top_video(
            cluster_df,
            cluster_id,
            date_str=os.path.basename(csv_path)
                .replace("comments_", "")
                .replace(".csv", ""),
            theme=theme,
        )

        if len(cluster_comments) < 20:
            continue

        weighted_titles = []
        if "video_title" in cluster_df.columns:
            video_stats = (
                cluster_df.groupby(["video_title", "channel_id"])
                .size().reset_index(name="comments")
                .sort_values("comments", ascending=False)
            )
            print("\nTop Videos")
            for _, row in video_stats.head(20).iterrows():
                info = CHANNEL_INFO.get(
                    row["channel_id"],
                    {"name": row["channel_id"], "ideology": "기타"}
                )
                print(f'{row["comments"]:>6,} | {row["video_title"]}')
                print(f'         Channel : {info["name"]}')
                print(f'         Ideology: {info["ideology"]}')
                print()
            print(f"Ideology: {ideology_text}")
            for title, count in cluster_df["video_title"].value_counts().items():
                weight = min(50, int(np.log1p(count) * 8))
                weighted_titles.extend([title] * weight)

        topic_docs = cluster_comments + weighted_titles
        try:
            topic_model = BERTopic(
                language="multilingual",
                calculate_probabilities=False,
                verbose=False
            )
            topics, _ = topic_model.fit_transform(topic_docs)
            topic_info = topic_model.get_topic_info()
            topic_info = topic_info[topic_info["Topic"] != -1]
            if len(topic_info) == 0:
                continue
            main_topic = topic_info.iloc[0]["Topic"]
            keywords = topic_model.get_topic(main_topic)
            title = " ".join([word for word, score in keywords[:5]])
            unique_titles = (
                cluster_df["video_title"].dropna().unique()
                if "video_title" in cluster_df.columns else []
            )
            event_titles[cluster_id] = title
            print(f"Cluster {cluster_id} ({len(cluster_comments):,} comments)")
            print(f"Videos: {len(unique_titles):,}")
            print(f"Title: {title}\n")
        except Exception as e:
            print(f"Cluster {cluster_id} failed: {e}")

    for cid in valid_clusters:
        print(f"Cluster {cid}: {cluster_counts[cid]:,} comments")

    print(f"\n유효 이벤트 수: {len(valid_clusters)}")

    # [v3 신규] event_title을 결과 csv 컬럼으로 추가
    event_df["event_title"] = event_df["event_id"].map(event_titles).fillna("")

    event_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    event_counts = pd.Series(labels).value_counts()
    print(f"  최대 클러스터 크기: {event_counts.max()}")
    print(f"  최소 클러스터 크기: {event_counts.min()}")
    print(f"  평균 클러스터 크기: {event_counts.mean():.1f}")
    return df


# ══════════════════════════════════════════════════════════════
# 8. 평가
# ══════════════════════════════════════════════════════════════

def run_evaluate(result_csv: str, label_col: str = 'event_id',
                 true_col: str = 'true_event_id'):
    df = pd.read_csv(result_csv)
    nmi, ami, ari = evaluate(df[true_col].tolist(), df[label_col].tolist())
    print(f'\n[평가] NMI={nmi:.4f}  AMI={ami:.4f}  ARI={ari:.4f}')
    return nmi, ami, ari


# ══════════════════════════════════════════════════════════════
# 9. 엔트리포인트
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python Hisevent.py 2021-12-14")
        sys.exit(1)

    date_str = sys.argv[1]
    CSV_PATH = resolve_csv_path(date_str, base_dir="spamremove_20260708_filtered")

    run(
        csv_path=CSV_PATH,
        language="Korean",
        n=100,
        max_num_neighbors=80,
        output_path="hisevent_results.csv"
    )