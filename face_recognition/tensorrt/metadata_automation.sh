python3 trt_mtcnn.py --video /face/tensorrt_demos/videos/example.mp4
python3 make_integral.py
curl -XPUT -u 'admin:Teampass1!' 'https://search-jetson-nano-uprffcac2sl2oj4yb6l4dhevnm.us-east-1.es.amazonaws.com/_bulk' --data-binary @result_integrated_3.json -H 'Content-Type: application/json'