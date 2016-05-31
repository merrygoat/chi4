import numpy as np
from scipy.spatial.distance import cdist
from time import time
from sys import argv, exit


def cdistmethod(data, numframes, averageparticles, dimensions, numberdensity, threshold):
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
<<<<<<< Updated upstream
    chisquared = chisquared/(numberdensity*averageparticles)

=======
    chisquared = chisquared*numberdensity*averageparticles
>>>>>>> Stashed changes
    return chisquared

def xyztocg(filename):

    inputfile = open(filename, 'r')

    outputfile = open(filename + ".cg", 'w')

    line = inputfile.readline()
    framenumber = 0
    particlenumber = 0

    while line != "":
        numparticles = int(line)  # read number of particles from first line of frame
        line = inputfile.readline()  # read comment from second line of frame

        for i in xrange(0, numparticles):
            line = inputfile.readline()
            outputchunk = line.split()
            outputfile.write(outputchunk[1] + "\t" +
                             outputchunk[2] + "\t" +
                             outputchunk[3].rstrip("\n") + "\t" +
                             str(framenumber) + "\t" +
                             str(particlenumber) + "\n")
            particlenumber += 1
        framenumber += 1
        line = inputfile.readline()  # read in number of particles in next frame
    outputfile.close()


def main():

<<<<<<< Updated upstream
    if len(argv) != 5:
        print("Incorrect syntax. Use Chi4.py filename.txt num_spatial_dimensions chi4_threshold number_density_in_sigma")
=======
    t = time()


    if len(argv) != 5:
        print("Incorrect syntax. Use Chi4.py filename.txt num_spatial_dimensions chi4_threshold number_density")
>>>>>>> Stashed changes
        exit()
    filename = argv[1]
    dimensions = int(argv[2])
    threshold = float(argv[3])
    numberdensity = float(argv[4])
<<<<<<< Updated upstream
=======
    zslices = 1

    if dimensions == 2:
        zslices = 1

    chisquaredresults = [[] for x in range(zslices)]

    for i in range(0, zslices):
        if zslices > 1:
            filename = "slice_" + i + "_ " + filename

        if filename.endswith("xyz"):
            xyztocg(filename)
            filename += ".cg"

        data = np.loadtxt(filename)
>>>>>>> Stashed changes

        numcolumns = data.shape[1]  # how many columns are there in the inut data?
        numframes = int(max(data[:, numcolumns - 2]) + 1)  # assume framenumber is penultimate column
        averageparticles = data.shape[0]/float(numframes)
        sortedmatrix = []

        for j in xrange(0, numframes):
            sortedmatrix.append(data[data[:, numcolumns - 2] == j, :])

        chisquaredresults[i] = cdistmethod(sortedmatrix, numframes, averageparticles, dimensions, numberdensity, threshold)

        np.savetxt(filename + "_chi4.txt", )

<<<<<<< Updated upstream
    a = time()
    np.savetxt(filename + "_chi4.txt", cdistmethod(sortedmatrix, numframes, averageparticles, dimensions, numberdensity, threshold))
    print time()-a
=======
    print time()-t
>>>>>>> Stashed changes

main()