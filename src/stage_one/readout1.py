# The function below is to simply return a dictionary with the readouts for stage 1. It uses all
# information calculated from previous steps to produce this.
# Arguments:
#   info: The dictionary that contains all the measurements we have done.
# Returns:
#   A dictionary containing all of the readouts for stage 1.
def readout(info):
    # Here, info['pathogenInfo']['area']'s length will be used to determine how many vacuoles
    # there are. info['pathogenInfo']['pathogens_in_vacuole'] will be used to determine
    # the number of pathogens in each vacuole.
    # info['cellInfo']['vacuole_number'] will be used to determine how many
    # cells there are.
    # Calculate % infected cells: n(infected)/n(non-infected)
    # Use info['cellInfo']['vacuole_number']
    vacNum = sum(info['cellInfo']['vacuole_number'])
    cellNum = len(info['cellInfo']['vacuole_number'])
    patNum = sum(info['pathogenInfo']['pathogens_in_vacuole'])
    
    percentInf = len([elem for elem in info['cellInfo']['vacuole_number'] if elem > 0])\
                    /cellNum if not cellNum == 0 else 0
    # Calculate Vacuole : Cells ratio
    vacCellRat = len(info['pathogenInfo']['area'])/cellNum if not cellNum == 0 else 0
    # Calculate pathogen load.
    patLoad = patNum/cellNum if not cellNum == 0 else 0
    # Calculate infection levels
    infectLevel = {
        '0': len([elem for elem in info['cellInfo']['vacuole_number'] if elem == 0])\
                    /cellNum if not cellNum == 0 else 0,
        '1': len([elem for elem in info['cellInfo']['vacuole_number'] if elem == 1])\
                    /cellNum if not cellNum == 0 else 0,
        '2': len([elem for elem in info['cellInfo']['vacuole_number'] if elem == 2])\
                    /cellNum if not cellNum == 0 else 0,
        '3': len([elem for elem in info['cellInfo']['vacuole_number'] if elem == 3])\
                    /cellNum if not cellNum == 0 else 0,
        '4': len([elem for elem in info['cellInfo']['vacuole_number'] if elem == 4])\
                    /cellNum if not cellNum == 0 else 0,
        '5': len([elem for elem in info['cellInfo']['vacuole_number'] if elem == 5])\
                    /cellNum if not cellNum == 0 else 0,
        '5+': len([elem for elem in info['cellInfo']['vacuole_number'] if elem > 5])\
                    /cellNum if not cellNum == 0 else 0,
    }
    # Calculate mean pathogen size
    meanPatSize = sum(info['pathogenInfo']['area'])/vacNum if not vacNum == 0 else 0
    # Calculate the mean vacuole position.
    vacPosition = sum(info['pathogenInfo']['dist_nuclear_centroid'])/vacNum if not vacNum == 0\
                  else 0
    # Calculate number of vacuoles that have replicating pathogens.
    percentRep = len([elem for elem in info['pathogenInfo']['pathogens_in_vacuole'] if elem > 1])\
                    /vacNum if not vacNum == 0 else 0
    # Calculate the replication distribution. I.e. how many vacuoles have one pathogen,
    # how many vacuoles have two pathogens, etc.
    repDist = {
        '1': len([elem for elem in info['pathogenInfo']['pathogens_in_vacuole'] if elem == 1])\
                    /vacNum if not vacNum == 0 else 0,
        '2': len([elem for elem in info['pathogenInfo']['pathogens_in_vacuole'] if elem == 2])\
                    /vacNum if not vacNum == 0 else 0,
        '4': len([elem for elem in info['pathogenInfo']['pathogens_in_vacuole'] if elem == 4])\
                    /vacNum if not vacNum == 0 else 0,
        '4+': len([elem for elem in info['pathogenInfo']['pathogens_in_vacuole'] if elem > 4])\
                    /vacNum if not vacNum == 0 else 0,
    }
    return {
        'percent_infected': percentInf,
        'vacuole_to_cell_ratio': vacCellRat,
        'pathogen_load': patLoad,
        'infection_levels': infectLevel,
        'mean_pathogen_size': meanPatSize,
        'vacuole_position': vacPosition,
        'percent_replicating_pathogens': percentRep,
        'replication_distribution': repDist
    }