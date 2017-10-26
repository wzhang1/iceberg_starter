python json_to_jpg.py --json_input train.json --output train --label_file train.lst --is_test False
python json_to_jpg.py --json_input test.json --output test --label_file test.lst --is_test True
python split.py 
python -u im2rec.py --resize 128 --quality 99 --num-thread 20 val ./
python -u im2rec.py --resize 128 --quality 99 --num-thread 20 train ./
python -u im2rec.py --resize 128 --quality 99 --num-thread 20 --list test.lst ./

