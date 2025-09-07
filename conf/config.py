

class CFG:
    wandb=False
    competition='FB3'
    _wandb_kernel='nakama'
    debug=False
    apex=True
    print_freq=20

    num_workers=0
    # Suggest: Rostlab/prot_bert or ESMs at https://huggingface.co/models?sort=downloads&search=esm
    # ESM examples: facebook/esm2_t33_650M_UR50D, facebook/esm1v_t33_650M_UR90S_1, facebook/esm1b_t33_650M_UR50S, etc 
    model="facebook/esm2_t33_650M_UR50D" 
    gradient_checkpointing=False
    scheduler='constant' # ['linear', 'cosine', 'constant']
    batch_scheduler=True
    num_warmup_steps=0
    
    # LEARNING RATE
    # Suggested for batchsize=8: Prot_bert = 5e-6, ESM2 = 5e-5
    epochs=1
    num_cycles=1.0 #only affects cosine schedule
    encoder_lr=5e-5
    decoder_lr=5e-5
    batch_size=1
    fast_debug=True
    debug_steps=2     # сколько батчей делаем в train_fn
    debug_epochs=1     # сколько эпох
    debug_val_steps =1
    debug_fold = 0
    
    # MODEL INFO - PROT_BERT
    total_blocks = 30 
    initial_layers = 5 
    layers_per_block = 16 
    # MODEL INFO - FACEBOOK ESM2
    if 'esm2' in model:
        total_blocks = int(model.split('_')[1][1:])
        initial_layers = 2 
        layers_per_block = 16 
        print('esm2 detected')
    # MODEL INFO - FACEBOOK ESM1B
    elif 'esm1b' in model:
        total_blocks = int(model.split('_')[1][1:])
        initial_layers = 4 
        layers_per_block = 16 
        print('esm1b detected')
    # MODEL INFO - FACEBOOK ESM1V
    elif 'esm1v' in model:
        total_blocks = int(model.split('_')[1][1:])
        initial_layers = 2 
        layers_per_block = 16 
        print('esm1v detected')
    else:
        print('prot_bert detected')
        
    # FREEZE
    # Suggested: Prot_bert -8, ESM2 -3
    num_freeze_blocks = total_blocks - 3
    
    # FOR NO FREEZE USE
    #num_freeze_blocks = 0
    
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    max_len=512
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    target_cols=['target']
    seed=42
    n_fold=5
    trn_fold=[0,1,2,3,4]
    train=True
    pca_dim = 64

    path='outputs/'
    config_path=path+'config.pth' 
    tokenizer_dir=path+'tokenizer/'
    
if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0]