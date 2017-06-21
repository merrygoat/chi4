import numpy as np
from scipy.spatial.distance import cdist
from time import time


def cdistmethod(data, numframes, averageparticles, dimensions, numberdensity, threshold):
    distance = np.zeros((numframes, 4))
    distancesquared = np.zeros((numframes, 3))

    for refframenumber in range(0, numframes):
        for curframenumber in range(refframenumber + 1, numframes):
            sqrtprod = cdist(data[refframenumber][:, 0:dimensions], data[curframenumber][:, 0:dimensions])
            numobservations = data[refframenumber][:, 0:dimensions].shape[0] * data[curframenumber][:, 0:dimensions].shape[0]

            result = float((np.sum(sqrtprod < threshold))) / np.sqrt(numobservations)
            distance[curframenumber - refframenumber, 0:2] += [result, 1]
            distancesquared[curframenumber - refframenumber, 0:2] += [(result * result), 1]

    # normalise by dividing by number of observations
    with np.errstate(divide='ignore', invalid='ignore'):  # suppress divide by zero warning since value is 0 at T=0
        distance[:, 2] = distance[:, 0] / distance[:, 1]
        distancesquared[:, 2] = distancesquared[:, 0] / distancesquared[:, 1]
    # square distance
    distance[:, 3] = distance[:, 2] ** 2

    chisquared = distancesquared[:, 2] - distance[:, 3]

    # normalise chi squared
    chisquared = chisquared * numberdensity * averageparticles
    return chisquared


def xyztocg(filename, simulationdata=1):
    inputfile = open(filename, 'r')

    outputfile = open(filename + ".cg", 'w')

    line = inputfile.readline()
    framenumber = 0
    particlenumber = 0

    while line != "":
        numparticles = int(line)  # read number of particles from first line of frame
        line = inputfile.readline()  # read comment from second line of frame

        for i in range(0, numparticles):
            line = inputfile.readline()
            outputchunk = line.split()
            outputfile.write(outputchunk[1] + "\t" + outputchunk[2] + "\t" + outputchunk[3].rstrip("\n") + "\t" 
                            + str(framenumber) + "\t" + str(particlenumber) + "\n")
            particlenumber += 1
        framenumber += 1
        if simulationdata == 1:
            particlenumber = 0
        line = inputfile.readline()  # read in number of particles in next frame
    outputfile.close()


def main(filename, dimensions, threshold, numberdensity):

    if filename.endswith("xyz"):
        xyztocg(filename)
        filename += ".cg"

    data = np.loadtxt(filename)

    numcolumns = data.shape[1]  # how many columns are there in the inut data?
    numframes = int(max(data[:, numcolumns - 2]) + 1)  # assume framenumber is penultimate column
    averageparticles = data.shape[0] / float(numframes)
    sortedmatrix = []

    for j in range(0, numframes):
        sortedmatrix.append(data[data[:, numcolumns - 2] == j, :])

    chisquaredresults = cdistmethod(sortedmatrix, numframes, averageparticles, dimensions, numberdensity, threshold)

    np.savetxt(filename + "_chi4.txt", chisquaredresults)


main()