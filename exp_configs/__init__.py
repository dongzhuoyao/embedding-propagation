from . import pretrain_exps, ssl_exps
from . import pretrain_exps
from . import pretrain_miniin_wrn_exps
from . import finetune_exps
from . import finetune_fast_exps

EXP_GROUPS = {}
EXP_GROUPS = pretrain_exps.EXP_GROUPS
EXP_GROUPS.update(ssl_exps.EXP_GROUPS)
EXP_GROUPS.update(pretrain_miniin_wrn_exps.EXP_GROUPS)
EXP_GROUPS.update(finetune_exps.EXP_GROUPS)
EXP_GROUPS.update(finetune_fast_exps.EXP_GROUPS)



