def RGB_calc(alpha):
    """This function willl take an alpha value [0,1]
    for a word and return an appropriate RGB value

    Will linearly go from 100,100,255 to 0,0,255 to 0,0,100
    """
    if alpha < 0.33:
        return (100-300*alpha, 100-300*alpha, 255)
    return (0, 0, 255-155*alpha)


if __name__ == '__main__':
    print(RGB_calc(0.14457932114601135))
    print(RGB_calc(0.2545487880706787))