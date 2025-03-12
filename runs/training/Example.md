🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
==((====))==  Unsloth 2025.2.15: Fast Llama patching. Transformers: 4.48.2.
   \\   /|    GPU: NVIDIA A100-SXM4-40GB. Max memory: 39.394 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.6.0+cu124. CUDA: 8.0. CUDA Toolkit: 12.4. Triton: 3.2.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.29.post2. FA2 = False]
 "-____-"     Free Apache license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Unsloth 2025.2.15 patched 32 layers with 32 QKV layers, 32 O layers and 32 MLP layers.
Generating train split: 100 examples [00:00, 2308.08 examples/s]
Dataset({
    features: ['instruction', 'input', 'output'],
    num_rows: 100
})
Map: 100%|██████████████████████████████████████████████████| 100/100 [00:00<00:00, 4355.00 examples/s]
Map (num_proc=2): 100%|██████████████████████████████████████| 100/100 [00:00<00:00, 133.94 examples/s]
GPU = NVIDIA A100-SXM4-40GB. Max memory = 39.394 GB.
5.748 GB of memory reserved.
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs = 1
   \\   /|    Num examples = 100 | Num Epochs = 5
O^O/ \_/ \    Batch size per device = 2 | Gradient Accumulation steps = 4
\        /    Total batch size = 8 | Total steps = 60
 "-____-"     Number of trainable parameters = 41,943,040
{'loss': 2.5045, 'grad_norm': 0.7697047591209412, 'learning_rate': 4e-05, 'epoch': 0.08}               
{'loss': 2.515, 'grad_norm': 0.801328718662262, 'learning_rate': 8e-05, 'epoch': 0.16}                 
{'loss': 2.5086, 'grad_norm': 0.7819052934646606, 'learning_rate': 0.00012, 'epoch': 0.24}             
{'loss': 2.4534, 'grad_norm': 0.814720869064331, 'learning_rate': 0.00016, 'epoch': 0.32}              
{'loss': 2.2574, 'grad_norm': 0.9636303782463074, 'learning_rate': 0.0002, 'epoch': 0.4}               
{'loss': 1.9708, 'grad_norm': 0.89924556016922, 'learning_rate': 0.00019636363636363636, 'epoch': 0.48}
{'loss': 1.6386, 'grad_norm': 0.9723618030548096, 'learning_rate': 0.00019272727272727274, 'epoch': 0.56}
{'loss': 1.3561, 'grad_norm': 1.287954330444336, 'learning_rate': 0.0001890909090909091, 'epoch': 0.64}
{'loss': 1.1037, 'grad_norm': 1.304614782333374, 'learning_rate': 0.00018545454545454545, 'epoch': 0.72}
{'loss': 0.8981, 'grad_norm': 0.8988156914710999, 'learning_rate': 0.00018181818181818183, 'epoch': 0.8}
{'loss': 0.7696, 'grad_norm': 0.999715268611908, 'learning_rate': 0.0001781818181818182, 'epoch': 0.88}
{'loss': 0.6995, 'grad_norm': 1.1495771408081055, 'learning_rate': 0.00017454545454545454, 'epoch': 0.96}
{'loss': 0.6271, 'grad_norm': 0.5324628949165344, 'learning_rate': 0.0001709090909090909, 'epoch': 1.0}
{'loss': 0.6069, 'grad_norm': 0.5186851024627686, 'learning_rate': 0.00016727272727272728, 'epoch': 1.08}
{'loss': 0.5944, 'grad_norm': 0.5191893577575684, 'learning_rate': 0.00016363636363636366, 'epoch': 1.16}
{'loss': 0.6245, 'grad_norm': 0.6574432253837585, 'learning_rate': 0.00016, 'epoch': 1.24}             
{'loss': 0.5793, 'grad_norm': 0.42645829916000366, 'learning_rate': 0.00015636363636363637, 'epoch': 1.32}
{'loss': 0.5896, 'grad_norm': 0.4369603097438812, 'learning_rate': 0.00015272727272727275, 'epoch': 1.4}
{'loss': 0.5772, 'grad_norm': 0.45507779717445374, 'learning_rate': 0.0001490909090909091, 'epoch': 1.48}
{'loss': 0.5661, 'grad_norm': 0.3147955536842346, 'learning_rate': 0.00014545454545454546, 'epoch': 1.56}
{'loss': 0.5633, 'grad_norm': 0.3215051293373108, 'learning_rate': 0.00014181818181818184, 'epoch': 1.64}
{'loss': 0.5474, 'grad_norm': 0.3700837790966034, 'learning_rate': 0.0001381818181818182, 'epoch': 1.72}
{'loss': 0.5445, 'grad_norm': 0.39912474155426025, 'learning_rate': 0.00013454545454545455, 'epoch': 1.8}
{'loss': 0.5492, 'grad_norm': 0.43788808584213257, 'learning_rate': 0.00013090909090909093, 'epoch': 1.88}
{'loss': 0.5451, 'grad_norm': 0.4315653443336487, 'learning_rate': 0.00012727272727272728, 'epoch': 1.96}
{'loss': 0.5311, 'grad_norm': 0.41650599241256714, 'learning_rate': 0.00012363636363636364, 'epoch': 2.0}
{'loss': 0.5332, 'grad_norm': 0.38589218258857727, 'learning_rate': 0.00012, 'epoch': 2.08}            
{'loss': 0.5204, 'grad_norm': 0.3183509409427643, 'learning_rate': 0.00011636363636363636, 'epoch': 2.16}
{'loss': 0.5077, 'grad_norm': 0.28283578157424927, 'learning_rate': 0.00011272727272727272, 'epoch': 2.24}
{'loss': 0.5146, 'grad_norm': 0.24624869227409363, 'learning_rate': 0.00010909090909090909, 'epoch': 2.32}
{'loss': 0.5102, 'grad_norm': 0.241265207529068, 'learning_rate': 0.00010545454545454545, 'epoch': 2.4}
{'loss': 0.5089, 'grad_norm': 0.2005520462989807, 'learning_rate': 0.00010181818181818181, 'epoch': 2.48}
{'loss': 0.5101, 'grad_norm': 0.23345156013965607, 'learning_rate': 9.818181818181818e-05, 'epoch': 2.56}
{'loss': 0.509, 'grad_norm': 0.2895093262195587, 'learning_rate': 9.454545454545455e-05, 'epoch': 2.64}
{'loss': 0.5072, 'grad_norm': 0.2903583347797394, 'learning_rate': 9.090909090909092e-05, 'epoch': 2.72}
{'loss': 0.4992, 'grad_norm': 0.27004629373550415, 'learning_rate': 8.727272727272727e-05, 'epoch': 2.8}
{'loss': 0.5147, 'grad_norm': 0.29760605096817017, 'learning_rate': 8.363636363636364e-05, 'epoch': 2.88}
{'loss': 0.5124, 'grad_norm': 0.2572937309741974, 'learning_rate': 8e-05, 'epoch': 2.96}               
{'loss': 0.4957, 'grad_norm': 0.29447805881500244, 'learning_rate': 7.636363636363637e-05, 'epoch': 3.0}
{'loss': 0.4943, 'grad_norm': 0.2370317578315735, 'learning_rate': 7.272727272727273e-05, 'epoch': 3.08}
{'loss': 0.5, 'grad_norm': 0.27176380157470703, 'learning_rate': 6.90909090909091e-05, 'epoch': 3.16}  
{'loss': 0.5024, 'grad_norm': 0.27183598279953003, 'learning_rate': 6.545454545454546e-05, 'epoch': 3.24}
{'loss': 0.4957, 'grad_norm': 0.23881135880947113, 'learning_rate': 6.181818181818182e-05, 'epoch': 3.32}
{'loss': 0.4829, 'grad_norm': 0.2525324821472168, 'learning_rate': 5.818181818181818e-05, 'epoch': 3.4}
{'loss': 0.4899, 'grad_norm': 0.24444745481014252, 'learning_rate': 5.4545454545454546e-05, 'epoch': 3.48}
{'loss': 0.4978, 'grad_norm': 0.33975857496261597, 'learning_rate': 5.090909090909091e-05, 'epoch': 3.56}
{'loss': 0.4821, 'grad_norm': 0.24126093089580536, 'learning_rate': 4.7272727272727275e-05, 'epoch': 3.64}
{'loss': 0.4964, 'grad_norm': 0.35511454939842224, 'learning_rate': 4.3636363636363636e-05, 'epoch': 3.72}
{'loss': 0.4977, 'grad_norm': 0.2111567258834839, 'learning_rate': 4e-05, 'epoch': 3.8}                
{'loss': 0.4918, 'grad_norm': 0.2246009260416031, 'learning_rate': 3.6363636363636364e-05, 'epoch': 3.88}
{'loss': 0.5016, 'grad_norm': 0.21434594690799713, 'learning_rate': 3.272727272727273e-05, 'epoch': 3.96}
{'loss': 0.4809, 'grad_norm': 0.3715123236179352, 'learning_rate': 2.909090909090909e-05, 'epoch': 4.0}
{'loss': 0.4877, 'grad_norm': 0.24743278324604034, 'learning_rate': 2.5454545454545454e-05, 'epoch': 4.08}
{'loss': 0.4855, 'grad_norm': 0.3792446255683899, 'learning_rate': 2.1818181818181818e-05, 'epoch': 4.16}
{'loss': 0.4814, 'grad_norm': 0.23675887286663055, 'learning_rate': 1.8181818181818182e-05, 'epoch': 4.24}
{'loss': 0.4921, 'grad_norm': 0.22217710316181183, 'learning_rate': 1.4545454545454545e-05, 'epoch': 4.32}
{'loss': 0.4764, 'grad_norm': 0.2494097352027893, 'learning_rate': 1.0909090909090909e-05, 'epoch': 4.4}
{'loss': 0.4813, 'grad_norm': 0.46723049879074097, 'learning_rate': 7.272727272727272e-06, 'epoch': 4.48}
{'loss': 0.4851, 'grad_norm': 0.3369899392127991, 'learning_rate': 3.636363636363636e-06, 'epoch': 4.56}
{'loss': 0.4784, 'grad_norm': 0.28980597853660583, 'learning_rate': 0.0, 'epoch': 4.64}                
{'train_runtime': 71.5487, 'train_samples_per_second': 6.709, 'train_steps_per_second': 0.839, 'train_loss': 0.7607600097854932, 'epoch': 4.64}
100%|██████████████████████████████████████████████████████████████████| 60/60 [01:11<00:00,  1.19s/it]
71.5487 seconds used for training.
1.19 minutes used for training.
Peak reserved memory = 6.576 GB.
Peak reserved memory for training = 0.828 GB.
Peak reserved memory % of max memory = 16.693 %.
Peak reserved memory for training % of max memory = 2.102 %.
<|begin_of_text|>Below is an instruction that describes a task, paired with an input along                   with its context. Write a response that appropriately completes the request.

### Instruction:
Can you provide a general overview of wildfire season in 2024, 2023, 2022, 2021, and          2020 in glacier national park?

### Input:


### Response:
The maximum annual temperature increase at grid R158C370 is projected to be 5.89899998°F by 2024. There is a 60.0% chance that grid R158C370 will experience wildfires by 2024. The maximum annual temperature increase at grid R158C370 is projected to be 6.80000019073569°F by 2023. There is a 70.0% chance that grid R158C370 will experience wildfires by 2023. The maximum annual temperature increase at grid R158C370 is projected to be 7.69999999°F by 2022
<|begin_of_text|>Below is an instruction that describes a task, paired with an input along                   with its context. Write a response that appropriately completes the request.

### Instruction:
What is the forecast for wildfire at Glacier National Park in August 2025

### Input:


### Response:
The maximum annual temperature increase at grid R153C311 is 5.63499984°F. There is a 50.0% chance that maximum annual temperature at this grid will exceed its historical range by end century. Grid R153C311 has a 20.0% chance of experiencing wildfires by end century.<|end_of_text|>
