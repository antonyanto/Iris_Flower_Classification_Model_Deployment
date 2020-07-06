from flask import Flask, render_template, request, url_for
import pickle

model = pickle.load(open("iris_classification_model.pkl",'rb'))

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method == "POST":
        sep_len = request.form['sepal-length']
        sep_wid = request.form['sepal-width']
        pet_len = request.form['petal-length']
        pet_wid = request.form['petal-width']
        result = model.predict([[sep_len,sep_wid,pet_len,pet_wid]])
        if result:
            return render_template("index.html",predicted= "Congratulations it's {res}".format(res=result[0]))
        else:
            return render_template("index.html",predicted="Something Went Wrong !")

if __name__ == "__main__":
    app.run(debug=True)