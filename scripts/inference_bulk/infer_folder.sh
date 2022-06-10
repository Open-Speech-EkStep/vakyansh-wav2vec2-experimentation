for i in $(ls -d /home/jupyter/stt_cleaned_data/train/*/); do f=$(sed 's#.*/##' <<< "${i%/}"); echo $f; bash prepare_data.sh $f ; bash infer.sh $f; done
