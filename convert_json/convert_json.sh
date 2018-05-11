CONVERT_JSON="/home/minhopark2115/git-reposits/Word2Vec_saramin/convert_json/"
STEP=$1
cp $CONVERT_JSON/graph_draw*.html ./
$CONVERT_JSON/convert_2_json.o nearest_topk_list_20_$STEP.csv nearest_topk_list_20_second_$STEP.csv
cp graph.json graph_20.json
$CONVERT_JSON/convert_2_json.o nearest_topk_list_02_$STEP.csv nearest_topk_list_02_second_$STEP.csv
mv graph.json graph_02.json
$CONVERT_JSON/convert_2_json.o nearest_topk_list_02_$STEP.csv nearest_topk_list_20_$STEP.csv nearest_topk_list_02_second_$STEP.csv nearest_topk_list_20_second_$STEP.csv





