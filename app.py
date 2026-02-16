from flask import Flask, jsonify, render_template
from simulation import TradingSimulation

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/simulate")
def simulate():
    sim = TradingSimulation()
    result = sim.run()
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
