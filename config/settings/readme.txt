pruning_type : server_pruning : 원래 하던 pruning. server-side에서 먼저 pruning한다는 의미에서 server pruning이라고 함.
pruning_type : local_pruning : local에서 먼저 pruning하고 server로 올려서 (distribution하기 전에) pruning 한다는 뜻에서 local pruning.


pruning: False  -- baseline 세팅

pruning : True + pruning_type -- experiment setting
