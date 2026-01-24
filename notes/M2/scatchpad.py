def snow_predictor(temperature: float = 20):
    if temperature < 0:
        print("Well, there's at least a chance!")
    else:
        print("Ain't no way, it's too warm!")

snow_predictor()
snow_predictor(-10)

