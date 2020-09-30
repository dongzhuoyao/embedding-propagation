from . import pretrain_exps, ssl_exps
from . import pretrain_exps
from . import pretrain_miniin_wrn_exps
from . import pretrain_miniin_wrn50_2_exps
from . import pretrain_miniin_resnet12_exps
from . import pretrain_miniin_resnet50_exps
from . import pretrain_miniin_densenet121_exps


from . import finetune_exps
from . import finetune_miniin_wrn_exps
from . import finetune_miniin_resnet12_exps



EXP_GROUPS = {}
EXP_GROUPS = pretrain_exps.EXP_GROUPS
EXP_GROUPS.update(ssl_exps.EXP_GROUPS)
EXP_GROUPS.update(pretrain_miniin_wrn_exps.EXP_GROUPS)
EXP_GROUPS.update(pretrain_miniin_wrn50_2_exps.EXP_GROUPS)
EXP_GROUPS.update(pretrain_miniin_resnet12_exps.EXP_GROUPS)
EXP_GROUPS.update(pretrain_miniin_resnet50_exps.EXP_GROUPS)
EXP_GROUPS.update(pretrain_miniin_densenet121_exps.EXP_GROUPS)


EXP_GROUPS.update(finetune_exps.EXP_GROUPS)
EXP_GROUPS.update(finetune_miniin_wrn_exps.EXP_GROUPS)
EXP_GROUPS.update(finetune_miniin_resnet12_exps.EXP_GROUPS)




