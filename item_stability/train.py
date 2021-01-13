from item_stability.lstm_keras_model import LSTMModel

if __name__ == "__main__":
    datadir = "/media/dmitriy/HDD/aruco_box_sim_02"
    model = LSTMModel(target="ang_v", epochs=100, target_frame=10, n_frames=3, datadir=datadir)
    model.train()
    model.save()
