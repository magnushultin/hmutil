import sys
'''
Settingsfile for circle detection

Name of tracker |   variable name  | example file
--------------------------------------------------
SMI Helmet          smi_helm          001.avi
Tobii Glasses       tobii             Data quality_Rec 35.avi
SMI Glasses         smi_glass         006-10-recording
ASL                 asl               120_00000.avi
PSTracker           ps_tracker        04-22-14_h10_m29_s52_edq_scene.mov
'''

def gauss(type):
    if type == 'smi_helm':
        return 5
    elif type == 'smi_glass':
        return 15
    elif type == 'asl':
        return 11
    elif type == 'ps_tracker':
        return 5
    elif type == 'tobii':
        return 11
    else:
        print 'error: no setting with that name'
        sys.exit(1)


def param1(type):
    if type == 'smi_helm':
        return 100
    elif type == 'smi_glass':
        return 100
    elif type == 'asl':
        return 100
    elif type == 'ps_tracker':
        return 100
    elif type == 'tobii':
        return 100
    else:
        print 'error: no setting with that name'
        sys.exit(1)

def param2(type):
    if type == 'smi_helm':
        return 10
    elif type == 'smi_glass':
        return 10
    elif type == 'asl':
        return 10
    elif type == 'ps_tracker':
        return 10
    elif type == 'tobii':
        return 10
    else:
        print 'error: no setting with that name'
        sys.exit(1)

def min_radius(type):
    if type == 'smi_helm':
        return 7
    elif type == 'smi_glass':
        return 7
    elif type == 'asl':
        return 7
    elif type == 'ps_tracker':
        return 7
    elif type == 'tobii':
        return 7
    else:
        print 'error: no setting with that name'
        sys.exit(1)

def max_radius(type):
    if type == 'smi_helm':
        return 14
    elif type == 'smi_glass':
        pass
    elif type == 'asl':
        pass
    elif type == 'ps_tracker':
        pass
    elif type == 'tobii':
        return 14
    else:
        print 'error: no setting with that name'
        sys.exit(1)

def min_area(type):
    if type == 'smi_helm':
        return 12 # 6
    elif type == 'smi_glass':
        return 200
    elif type == 'asl':
        return 20
    elif type == 'ps_tracker':
        return 6
    elif type == 'tobii':
        return 34
    else:
        print 'error: no setting with that name'
        sys.exit(1)

def max_area(type):
    if type == 'smi_helm':
        return 50
    elif type == 'smi_glass':
        return 300
    elif type == 'asl':
        return 80
    elif type == 'ps_tracker':
        return 100
    elif type == 'tobii':
        return 106
    else:
        print 'error: no setting with that name'
        sys.exit(1)

def fix_treshold(type):
    if type == 'smi_helm':
        return 190
    elif type == 'smi_glass':
        return 0
    elif type == 'asl':
        return 180
    elif type == 'ps_tracker':
        return 140
    elif type == 'tobii':
        return 170
    else:
        print 'error: no setting with that name'
        sys.exit(1)

def size_corr(type):
    if type == 'smi_helm':
        return 0
    elif type == 'smi_glass':
        return 0
    elif type == 'asl':
        return 0
    elif type == 'ps_tracker':
        return 0
    elif type == 'tobii':
        return 8
    else:
        print 'error: no setting with that name'
        sys.exit(1)

def roundness(type):
    if type == 'smi_helm':
        return 0.7
    elif type == 'smi_glass':
        return 0.7
    elif type == 'asl':
        return 0.7
    elif type == 'ps_tracker':
        return 0.7
    elif type == 'tobii':
        return 0.7
    else:
        print 'error: no setting with that name'
        sys.exit(1)
