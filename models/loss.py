import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureContrastiveLoss(nn.Module):
    """
    Satellite-view texture contrastive loss
    Intra-class compactness + inter-class separation (texture features only)
    """
    def __init__(self, num_semantics=4, margin=1.0):
        super().__init__()
        self.K = num_semantics
        self.margin = margin

    def forward(self, F_tex, S):
        """
        F_tex: (B, C, H, W) satellite native texture features
        S:     (B, H, W)     satellite semantic mask
        """
        B, C, H, W = F_tex.shape
        device = F_tex.device

        # 1. Flatten features
        F_flat = F_tex.flatten(2)          # (B, C, N)
        S_flat = S.flatten(1)              # (B, N)

        # 2. One-hot encoding
        S_onehot = F.one_hot(S_flat, num_classes=self.K).float()  # (B, N, K)

        # 3. Compute class means
        class_counts = S_onehot.sum(dim=1) + 1e-8  # (B, K)
        F_by_class = torch.bmm(S_onehot.transpose(1, 2), F_flat.transpose(1, 2))
        F_class_mean = F_by_class / class_counts.unsqueeze(-1)  # (B, K, C)

        # 4. Intra-class loss (variance)
        F_mean_expanded = F_class_mean.unsqueeze(2)  # (B, K, 1, C)
        F_flat_expanded = F_flat.transpose(1, 2).unsqueeze(1)  # (B, 1, N, C)
        diff = (F_flat_expanded - F_mean_expanded) ** 2
        intra_loss = (S_onehot.unsqueeze(-1) * diff).sum(dim=(2, 3))
        intra_loss = intra_loss / class_counts
        valid_mask = (class_counts > 1).float()
        loss_intra = (intra_loss * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)
        loss_intra = loss_intra.mean()

        # 5. Inter-class loss (distance)
        dist_matrix = torch.cdist(F_class_mean, F_class_mean, p=2)  # (B, K, K)
        triu_mask = torch.triu(torch.ones(self.K, self.K), diagonal=1).bool().to(device)
        valid_pair = valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(-2) & triu_mask.unsqueeze(0)
        loss_inter = torch.relu(self.margin - dist_matrix) * valid_pair.float()
        loss_inter = loss_inter.sum(dim=(1, 2)) / (valid_pair.sum(dim=(1, 2)) + 1e-8)
        loss_inter = loss_inter.mean()

        # 6. Total loss
        loss_texture = loss_intra + loss_inter

        return loss_texture, loss_intra, loss_inter


# =============================================================================
# SS-CVNet loss functions
# =============================================================================

class GeoViewNetLoss(nn.Module):
    """
    SS-CVNet loss function

    Combined loss:
    1. Base MSE loss: measures the difference between predictions and ground truth
    2. Physics constraint loss: ensures BVI + GVI + SVF <= 1

    Formulas:
        L_base = (1/3) * Σ(ŷ_i - y_i)²  for i ∈ {BVI, GVI, SVF}
        L_physics = ReLU(ŷ_BVI + ŷ_GVI + ŷ_SVF - 1)²
        L_total = L_base + λ * L_physics

    where λ = 0.1
    """

    def __init__(self, lambda_physics: float = 0.1):
        """
        Args:
            lambda_physics: weight coefficient of the physics constraint loss (default 0.1)
        """
        super(GeoViewNetLoss, self).__init__()
        self.lambda_physics = lambda_physics
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> tuple:
        """
        Compute the total loss

        Args:
            predictions: predictions [B, 3], in the format [BVI, GVI, SVF]
            targets: ground truth [B, 3], in the format [BVI, GVI, SVF]

        Returns:
            tuple: (total_loss, base_loss, physics_loss)
                - total_loss: total loss
                - base_loss: base MSE loss
                - physics_loss: physics constraint loss
        """
        # 1. Base MSE loss
        # L_base = (1/3) * Σ(ŷ_i - y_i)²
        base_loss = self.mse_loss(predictions, targets)

        # 2. Physics constraint loss
        # L_physics = ReLU(ŷ_BVI + ŷ_GVI + ŷ_SVF - 1)²
        # The sum of predictions should be <= 1 (physics constraint: the sum of the three indicators does not exceed 1)
        pred_sum = predictions.sum(dim=1)  # [B]
        constraint_violation = F.relu(pred_sum - 1.0)  # [B], non-zero only when the sum > 1
        physics_loss = (constraint_violation ** 2).mean()  # scalar

        # 3. Total loss
        # L_total = L_base + λ * L_physics
        total_loss = base_loss + self.lambda_physics * physics_loss

        return total_loss, base_loss, physics_loss

    def compute_base_loss_only(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute only the base MSE loss (for debugging or ablation experiments)

        Args:
            predictions: predictions [B, 3]
            targets: ground truth [B, 3]

        Returns:
            base_loss: base MSE loss
        """
        return self.mse_loss(predictions, targets)

    def compute_physics_loss_only(
        self,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute only the physics constraint loss (for debugging or ablation experiments)

        Args:
            predictions: predictions [B, 3]

        Returns:
            physics_loss: physics constraint loss
        """
        pred_sum = predictions.sum(dim=1)
        constraint_violation = F.relu(pred_sum - 1.0)
        physics_loss = (constraint_violation ** 2).mean()
        return physics_loss


class GeoViewNetLossWithLogging(GeoViewNetLoss):
    """
    SS-CVNet loss function with logging

    Inherits from GeoViewNetLoss and adds:
    1. Detailed loss statistics
    2. Constraint violation rate monitoring
    3. Independent MSE loss for each indicator
    """

    def __init__(self, lambda_physics: float = 0.1, log_interval: int = 100):
        """
        Args:
            lambda_physics: weight coefficient of the physics constraint loss (default 0.1)
            log_interval: logging interval (steps)
        """
        super().__init__(lambda_physics)
        self.log_interval = log_interval
        self.step_count = 0

        # Statistics
        self.stats = {
            'total_loss': [],
            'base_loss': [],
            'physics_loss': [],
            'constraint_violation_rate': [],
            'bvi_mse': [],
            'gvi_mse': [],
            'svf_mse': []
        }

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> tuple:
        """
        Compute the loss and record statistics

        Args:
            predictions: predictions [B, 3]
            targets: ground truth [B, 3]

        Returns:
            tuple: (total_loss, base_loss, physics_loss, info_dict)
                - info_dict: dict containing detailed statistics
        """
        # Call the parent method to compute the loss
        total_loss, base_loss, physics_loss = super().forward(predictions, targets)

        # Compute statistics
        with torch.no_grad():
            # 1. Constraint violation rate
            pred_sum = predictions.sum(dim=1)
            violated = (pred_sum > 1.0).float().sum().item()
            violation_rate = violated / predictions.size(0)

            # 2. Independent MSE loss for each indicator
            bvi_mse = F.mse_loss(predictions[:, 0], targets[:, 0]).item()
            gvi_mse = F.mse_loss(predictions[:, 1], targets[:, 1]).item()
            svf_mse = F.mse_loss(predictions[:, 2], targets[:, 2]).item()

            # 3. Update statistics
            self.step_count += 1
            if self.step_count % self.log_interval == 0:
                self.stats['total_loss'].append(total_loss.item())
                self.stats['base_loss'].append(base_loss.item())
                self.stats['physics_loss'].append(physics_loss.item())
                self.stats['constraint_violation_rate'].append(violation_rate)
                self.stats['bvi_mse'].append(bvi_mse)
                self.stats['gvi_mse'].append(gvi_mse)
                self.stats['svf_mse'].append(svf_mse)

        # 4. Build the info dict
        info_dict = {
            'total_loss': total_loss.item(),
            'base_loss': base_loss.item(),
            'physics_loss': physics_loss.item(),
            'violation_rate': violation_rate,
            'bvi_mse': bvi_mse,
            'gvi_mse': gvi_mse,
            'svf_mse': svf_mse,
            'pred_sum_mean': pred_sum.mean().item(),
            'pred_sum_max': pred_sum.max().item(),
            'pred_sum_min': pred_sum.min().item()
        }

        return total_loss, base_loss, physics_loss, info_dict

    def get_stats_summary(self) -> dict:
        """
        Get a summary of the statistics

        Returns:
            dict: statistics summary (mean, std, etc.)
        """
        summary = {}
        for key, values in self.stats.items():
            if len(values) > 0:
                values_tensor = torch.tensor(values)
                summary[key] = {
                    'mean': values_tensor.mean().item(),
                    'std': values_tensor.std().item(),
                    'min': values_tensor.min().item(),
                    'max': values_tensor.max().item(),
                    'count': len(values)
                }
            else:
                summary[key] = None
        return summary

    def reset_stats(self):
        """Reset statistics"""
        for key in self.stats:
            self.stats[key] = []
        self.step_count = 0


def create_geoview_loss(
    lambda_physics: float = 0.1,
    with_logging: bool = False,
    log_interval: int = 100
) -> nn.Module:
    """
    Factory function: create the SS-CVNet loss function

    Args:
        lambda_physics: weight coefficient of the physics constraint loss (default 0.1)
        with_logging: whether to use the version with logging (default False)
        log_interval: logging interval (default 100 steps)

    Returns:
        loss_fn: loss function instance

    Examples:
        >>> # Basic version
        >>> loss_fn = create_geoview_loss(lambda_physics=0.1)
        >>> total_loss, base_loss, physics_loss = loss_fn(pred, target)

        >>> # Version with logging
        >>> loss_fn = create_geoview_loss(with_logging=True, log_interval=50)
        >>> total_loss, base_loss, physics_loss, info = loss_fn(pred, target)
        >>> print(f"Violation rate: {info['violation_rate']:.2%}")
    """
    if with_logging:
        return GeoViewNetLossWithLogging(
            lambda_physics=lambda_physics,
            log_interval=log_interval
        )
    else:
        return GeoViewNetLoss(lambda_physics=lambda_physics)
