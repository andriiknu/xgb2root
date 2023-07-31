from xgboost import XGBClassifier
import ROOT
agc_models_dir = './models'


def convert_models():
    model_even = XGBClassifier()
    model_even.load_model(f'{agc_models_dir}/model_even.json')
    # model_even.max_depth = 10

    model_odd = XGBClassifier()
    model_odd.load_model(f'{agc_models_dir}/model_odd.json')
    # model_even.max_depth = 10

    ROOT.TMVA.Experimental.SaveXGBoost(
        model_even, 
        'agc_even', 
        'models/model_even.root',
        # num_inputs=100 
    )

    ROOT.TMVA.Experimental.SaveXGBoost(
        model_odd, 
        'agc_odd', 
        'models/model_odd.root', 
        # num_inputs=100 
    )

if __name__ == "__main__":
    convert_models()