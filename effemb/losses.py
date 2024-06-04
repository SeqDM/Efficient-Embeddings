import gin
import torch
import torch.nn as nn


# GTE paper
@gin.configurable
def all_possible_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    tau: torch.Tensor,
    cross_entropy_loss: nn.CrossEntropyLoss,
    negatives: torch.Tensor = None,
) -> torch.Tensor:
        # Assuming that the output_tensor is of shape (batch_size, hidden_size)
        # We want batch contrastive loss
        # Both should be of the shape B x d
        assert x.shape == y.shape
        x = x.div(x.norm(dim=-1, keepdim=True));
        y = y.div(y.norm(dim=-1, keepdim=True));
        y_y_sim = y.mm(y.T)
        y_x_sim = y.mm(x.T)
        x_y_sim = y_x_sim.T
        x_x_sim = x.mm(x.T)
        Z_1 = y_y_sim.mul(tau).exp() + x_x_sim.mul(tau).exp()
        Z_2 = y_x_sim.mul(tau).exp() + x_y_sim.mul(tau).exp()
        Z_1[range(x.shape[0]), range(x.shape[0])] = 0
        Z = Z_1 + Z_2
        Z = Z.sum(dim=-1)
        L_icl = -y_x_sim.diag().mul(tau).exp().div(Z).log().mean()
        return L_icl


@gin.configurable
def double_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    tau: torch.Tensor,
    cross_entropy_loss: nn.CrossEntropyLoss,
    negatives: torch.Tensor = None,
) -> torch.Tensor:
        # Assuming that the output_tensor is of shape (batch_size, hidden_size)
        # We want batch contrastive loss
        # Both should be of the shape B x d
        assert x.shape == y.shape
        normalised_x = nn.functional.normalize(x, p=2, dim=1)
        normalised_y = nn.functional.normalize(y, p=2, dim=1)
        logits = torch.matmul(normalised_x, normalised_y.T) * tau
        batch_size = x.size(0)
        labels = torch.arange(batch_size)
        loss = (
            single_loss(x, y, tau, cross_entropy_loss, negatives) + cross_entropy_loss(logits.T, labels)
        ) / 2
        return loss


@gin.configurable
def single_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    tau: torch.Tensor,
    cross_entropy_loss: nn.CrossEntropyLoss,
    negatives: torch.Tensor = None,
) -> torch.Tensor:
        # Assuming that the output_tensor is of shape (batch_size, hidden_size)
        # We want batch contrastive loss
        # Both should be of the shape B x d
        assert x.shape == y.shape

        if negatives is not None:
            y = torch.cat((y, negatives))

        normalised_x = nn.functional.normalize(x, p=2, dim=1)
        normalised_y = nn.functional.normalize(y, p=2, dim=1)
        logits = torch.matmul(normalised_x, normalised_y.T) * tau
        batch_size = x.size(0)
        labels = torch.arange(batch_size)

        return cross_entropy_loss(logits, labels)
