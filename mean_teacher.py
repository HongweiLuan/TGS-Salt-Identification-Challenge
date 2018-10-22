from common import *

#################################################################################################################
#
# [1] 'Self-ensembling for visual domain adaptation' - Geoffrey French, Michal Mackiewicz, Mark Fisher, arvix 2018
#        https://arxiv.org/abs/1706.05208
#        https://github.com/Britefury/self-ensemble-visual-domain-adapt
#
# [2] 'Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning
# results' - Antti Tarvainen and Harri Valpola. 2017.
#        https://arxiv.org/abs/1703.01780
#        https://github.com/CuriousAI/mean-teacher


def detatch_teacher_parameters(net_teacher):
    for param in net_teacher.parameters():
        #param.requires_grad = False
        param.detach_()

def zero_teacher_parameters(net_teacher):
    for param in net_teacher.parameters():
        param.zero_()


def update_teacher_parameters(net_student, net_teacher, alpha_student=0.5, alpha_teacher=0.5):

    # net_teacher.load_state_dict(net_student.state_dict())
    # return
    for param_teacher, param_student in zip(net_teacher.parameters(), net_student.parameters()):
        param_teacher.data = alpha_teacher*param_teacher.data + alpha_student*param_student.data





## Exponential rampup from https://arxiv.org/abs/1610.02242
def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))



def criterion_unlabel(logit, target):
    assert logit.size() == target.size()

    batch_size = len(logit)
    logit  = logit.view(batch_size,-1)
    target = target.detach().view(batch_size,-1)
    prob   = F.sigmoid(logit)
    target = F.sigmoid(target)

    loss = F.mse_loss(prob, target, size_average=True)  #/batch_size

    return loss



