import torch


def test_prefill_selection_chunks_score_slab_without_changing_result(monkeypatch):
    import freetoken.attention.dsv4_indexer as module
    from freetoken.attention.dsv4_indexer import IndexerBackendMixin

    class Backend(IndexerBackendMixin):
        chunks = []

        def indexer_prefill_logits(self, q, keys, weights):
            self.chunks.append(q.shape[1])
            # Every query ranks later blocks higher; the causal mask determines visibility.
            return torch.arange(keys.shape[1], dtype=torch.float32).view(1, 1, -1).expand(
                q.shape[0], q.shape[1], -1
            ).clone()

    # Four fp32 columns make each score row 16 B; force two query rows per score slab.
    monkeypatch.setattr(module, "_PREFILL_SCORE_BYTES", 32)
    backend = Backend()
    q = torch.empty(1, 5, 1, 1)
    keys = torch.empty(1, 4, 1)
    weights = torch.empty(1, 5, 1)

    selected = backend.indexer_prefill_select(
        q, keys, weights, start_pos=2, seqlen=5, ratio=1, topk=2, offset=0
    )

    assert backend.chunks == [2, 2, 1]
    assert selected.tolist() == [[[2, 1], [3, 2], [3, 2], [3, 2], [3, 2]]]
