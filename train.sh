if [ "$1" = "" -o "$2" = "" ]; then
    echo "Usage: ./train.sh <ray number (batch size)> <dataset name>"
    echo "For example: ./train.sh 1024 car"
    if [ "a"$1 = "a" ]; then
        echo "Please specify batch size (ray number)"
    fi
    if [ "a"$2 = "a" ]; then
        echo "Please specify dataset name (object name, for example: lego)"
    fi
    exit
fi

python3 ./train.py -s -t -u --sample_ray_num $1 --dataset_name $2 --render_depth --render_normal