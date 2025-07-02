import torch.nn as nn
import torch.nn.functional as F

class DINOLoss(nn.Module):
    def __init__(self, out_dim, temp_student=0.1, temp_teacher=0.07, momentum=0.996):
        super().__init__()
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
        self.momentum = momentum
        self.out_dim = out_dim

        # Optional logging support
        self.writer = None
        self.global_step = 0

    def forward(self, student_outs, teacher_outs):
        loss = 0
        for i, t_out in enumerate(teacher_outs):
            t_probs = F.softmax(t_out / self.temp_teacher, dim=-1).detach()
            for j, s_out in enumerate(student_outs):
                s_log_probs = F.log_softmax(s_out / self.temp_student, dim=-1)
                pair_loss = -(t_probs * s_log_probs).sum(dim=-1).mean()
                loss += pair_loss

                # Log individual pair loss
                if self.writer is not None:
                    self.writer.add_scalar(f"Loss/pair_t{i}_s{j}", pair_loss.item(), self.global_step)

        loss /= (len(teacher_outs) * len(student_outs))
        return loss
