from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils import common_functions as c_f


class SimilarityTracker:
    def __init__(
        self,
        writer=None,
        distance=None
    ):
        self.distance = distance
        self.writer = writer
        self.step = 0

    def __call__(
        self, embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None
    ):
        c_f.check_shapes(embeddings, labels)
        labels = c_f.to_device(labels, embeddings)
        ref_emb, ref_labels = c_f.set_ref_emb(
            embeddings, labels, ref_emb, ref_labels)
        self.compute_loss(
            embeddings, labels, indices_tuple, ref_emb, ref_labels
        )

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, ref_labels, t_per_anchor="all"
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple

        mat = self.distance(embeddings, ref_emb)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]

        self.writer.add_scalar('Similarity/Positive',
                               ap_dists.detach().mean().cpu(), self.step)
        self.writer.add_scalar('Similarity/Negative',
                               an_dists.detach().mean().cpu(), self.step)
        self.writer.add_scalar('Similarity/Difference',
                               (an_dists.detach().mean() - ap_dists.detach().mean()).cpu(), self.step)
        self.step += 1
