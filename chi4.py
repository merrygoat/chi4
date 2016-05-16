import numpy as np
from scipy.spatial.distance import cdist
from time import time
from sys import argv, exit


def cdistmethod(data, numframes, numparticles, dimensions, numberdensity, temperature, threshold):
    distance = np.zeros((numframes, 4))
    distancesquared = np.zeros((numframes, 3))

    for refframenumber in xrange(0, numframes):
        for curframenumber in xrange(refframenumber+1, numframes):
            sqrtprod = cdist(data[refframenumber][:, 0:dimensions], data[curframenumber][:, 0:dimensions])
            numobservations = data[refframenumber][:, 0:dimensions].shape[0] * data[curframenumber][:, 0:dimensions].shape[0]

            result = float((np.sum(sqrtprod < threshold)))/np.sqrt(numobservations)
            distance[curframenumber-refframenumber, 0:2] += [result, 1]
            distancesquared[curframenumber-refframenumber, 0:2] += [(result*result), 1]

    # normalise by dividing by number of observations
    with np.errstate(divide='ignore', invalid='ignore'):  # suppress divide by zero warning since value is 0 at T=0
        distance[:, 2] = distance[:, 0]/distance[:, 1]
        distancesquared[:, 2] = distancesquared[:, 0]/distancesquared[:, 1]
    # square distance
    distance[:, 3] = distance[:, 2]**2

    chisquared = distancesquared[:, 2] - distance[:, 3]

    # normalise chi squared
    chisquared = chisquared*numberdensity*numparticles*temperature

    return chisquared

def xyztocg(filename):

    inputfile = open(filename, 'r')
    data = inputfile.readlines()
    inputfile.close()

    numparticles = int(data[0])
    numlines = len(data)

    outputfile = open(filename + ".cg", 'w')

    for i in range(0, numlines):
        if i % (numparticles+2) == 0:
            # num particles line
            pass
        elif i % (numparticles+2) == 1:
            # comment line
            pass
        else:
            outputchunk = data[i].split()
            outputfile.write(outputchunk[1] + "\t" +
                             outputchunk[2] + "\t" +
                             outputchunk[3].rstrip("\n") + "\t" +
                             str(i//(numparticles+2)) + "\t" +
                             str((i % (numparticles+2))-2) + "\n")

    outputfile.close()


def main():

    if len(argv) != 6:
        print("Incorrect syntax. Use Chi4.py filename.txt num_spatial_dimensions chi4_threshold number_density temperature")
        exit()
    filename = argv[1]
    dimensions = int(argv[2])
    threshold = float(argv[3])
    numberdensity = float(argv[4])
    temperature = float(argv[5])

    if filename.endswith("xyz"):
        xyztocg(filename)
        filename += ".cg"

    data = np.loadtxt(filename, delimiter="\t")

    numframes = int(max(data[:, dimensions]) + 1)
    maxnumparticles = 0

    for i in xrange(0, numframes):
        particlesinframe = data[data[:,dimensions] == i, :].shape[0]
        if particlesinframe > maxnumparticles:
            maxnumparticles = particlesinframe

    sortedmatrix = []

    for i in xrange(0, numframes):
        sortedmatrix.append(data[data[:, dimensions] == i, :])

    a = time()
    np.savetxt(filename + "_chi4.txt", cdistmethod(sortedmatrix, numframes, maxnumparticles, dimensions, numberdensity, temperature, threshold))
    print time()-a

main()