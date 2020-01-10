eos=$1
del=$2
decode_from_src=$3
path=levt-seperate-final-layer
modelpath=checkpoints/gec/levt-separate-final-layer-old-data
is_levt=True

eos_str=${eos//[.]/_}
del_str=${del//[.]/_}
decode_mode=`if [ $decode_from_src == "True" ]; then printf %s '.from-src' ; fi`
ext=$decode_mode.eos-$eos_str.del-$del_str

mkdir -p results/$path;

if [ $is_levt = "True" ]; then
   python3 generate.py \
    data-bin/gec-small-finetune \
    --gen-subset test \
    --task translation_lev \
    --path $modelpath/checkpoint_best.pt \
    --iter-decode-max-iter 10 \
    --iter-decode-eos-penalty $eos \
    --iter-decode-del-penalty $del \
    `if [ $decode_from_src == "True" ]; then printf %s '--decode-from-source' ; fi` \
    --beam 5 --remove-bpe \
    --retain-iter-history\
    --print-step \
    --sacrebleu \
    --max-tokens 8000 > results/$path/gec$ext.out
else
  python3 generate.py \
    data-bin/gec-new-finetune \
    --gen-subset test \
    --path $modelpath/checkpoint_best.pt \
    --beam 5 --remove-bpe \
    --sacrebleu \
    --max-tokens 8000 > results/$path/gec$ext.out
fi
    


cd results/$path    

sort -V gec$ext.out > gec$ext.sorted

grep ^H gec$ext.sorted | cut -f3- > gec$ext.cor
rm gec$ext.sorted

python3 ../../score.py --sys gec$ext.cor --ref ../test.cor | tee score$ext.bleu
python3 ../../../errant/parallel_to_m2.py -orig ../original.wrg -cor gec$ext.cor -out gec$ext.m2
python3 ../../../errant/compare_m2.py -hyp gec$ext.m2 -ref ../official-2014.combined.m2  | tee score$ext.m2

rm gec$ext.m2
cd ../..
