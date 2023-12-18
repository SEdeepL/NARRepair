# NARRepair
The code of paper "NARRepair:Non-Autoregressive Code Generation Model for Automatic Program Repair"
### Requirements
* Python >= 3.7
* Pytorch >= 1.5.0
* Fairseq >=1.0.0
* Tree-Sitter
* Transformers>=4.10.0
## Folder Structure
 ```bash
 ├── Dataset: data used for training and testing
 ├── fairseq: the code of fairseq from
     ├──models/nat/narrepair_nonautoregressive_transformer.py: the code of NARRepair model
     ├──parser: the code of generating AST using Tree-Sitter tool
 ├── narrepair: the code of NARRepair
     ├──narrepair/task: the code of task of NARRepair
     ├──narrepair/model: the code of NARRepair model
     ├──narrepair/criterions: the code of criterions function of NARRepair

```
### Preprocess
```
TEXT=
dict_path1=

python $EXP_HOME/preprocess.py --source-lang buggy  --target-lang fixed   \
    --task translation \
    --trainpref $TEXT/train --validpref $TEXT/valid \
    --padding-factor 8 \
    --src-with-werdur \
    --destdir defect4j-bin \
    --srcdict ${dict_path1} --tgtdict ${dict_path1} \
    --workers 60
```
### Train
```
data_dir=
save_path=
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py ${data_dir} --arch narrepair --noise full_mask --share-all-embeddings \
    --criterion narrepair_loss --label-smoothing 0.1 --lr 5e-5 --warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-6 --task narrepair_task --max-tokens 50000 --weight-decay 0.01 --dropout 0.1 \
    --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 \
    --max-source-positions 6000 --max-target-positions 6000 --seed 0 --clip-norm 5 \
    --save-dir ${save_path} --src-embedding-copy --length-loss-factor 0.05 --log-interval 100 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu --user-dir narrepair --mlm-layers 2 --batch-size 50 --max-epoch 100 \
    --src-with-werdur --werdur-max-predict 10
```

### Inference
```
checkpoint_path=
data_dir=
src=buggy
tgt=fixed
CUDA_VISIBLE_DEVICES=0 python3 fairseq_cli/generate.py ${data_dir} --path ${checkpoint_path} \
 --user-dir narrepair --task narrepair_task --remove-bpe --source-lang ${src} \
 --target-lang ${tgt} --max-sentences 20  --iter-decode-max-iter 0 --iter-decode-force-max-iter \
 --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 --gen-subset test \
```
