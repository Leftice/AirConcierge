set -e # stop script when there is an error

if [ "$1" = "air" ]
then
    echo "Preprocessing AirDialogue dataset !"
    cd parser
    bash parser_run_air.sh
    cd ..
elif [ "$1" = "syn" ]
then
    echo "Preprocessing Synthesized dataset !"
    cd parser
    bash parser_run_syn.sh
    cd ..
else
    echo "Please choose dataset to processed ex : air or syn"
fi
