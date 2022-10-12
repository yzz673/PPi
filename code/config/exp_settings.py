# 5-1-1 setting
class ExpSettings(object):
    def __init__(self):
        p = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6']
        self.exp_ids = {
            'an exp id': {'source': {'filenames': [p[0], p[1], p[2], p[3], p[4]},
                          'valid':  {'filenames': [p[5]]},
                          'target': {'filenames': [p[6]]}
                        },  # just a demo
        }
