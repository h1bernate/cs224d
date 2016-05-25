echo "Please enter yout sunetid: "
read sunetid

zip -r $sunetid.zip rnn.py weights loss_history*.png rnn_tuner_*.py
