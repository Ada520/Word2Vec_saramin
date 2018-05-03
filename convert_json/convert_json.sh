CONVERT_JSON="/home/minhopark2115/git-reposits/Word2Vec_saramin/convert_json/"
cp $CONVERT_JSON/graph_draw*.html ./
$CONVERT_JSON/convert_2_json.o nearest_topk_list_20_2000000.csv nearest_topk_list_20_second_2000000.csv
cp graph.json graph_20.json
$CONVERT_JSON/convert_2_json.o nearest_topk_list_02_2000000.csv nearest_topk_list_02_second_2000000.csv
mv graph.json graph_02.json
$CONVERT_JSON/convert_2_json.o nearest_topk_list_02_2000000.csv nearest_topk_list_20_2000000.csv nearest_topk_list_02_second_2000000.csv nearest_topk_list_20_second_2000000.csv





