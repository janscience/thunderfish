from nose.tools import assert_true, with_setup
import os
import numpy as np
import thunderfish.datawriter as dw
import thunderfish.dataloader as dl
from audioio import flatten_metadata


relacs_path = 'test_relacs'
fishgrid_path = 'test_fishgrid'


def generate_data():
    samplerate = 44100.0
    duration = 100.0
    channels = 4
    t = np.arange(int(duration*samplerate))/samplerate
    data = np.sin(2.0*np.pi*880.0*t) * t/duration
    data = data.reshape((-1, 1))
    for k in range(data.shape[1], channels):
        data = np.hstack((data, data[:,0].reshape((-1, 1))/k))
    return data, samplerate

    
def remove_files(path):
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
        os.rmdir(os.path.join(path))

        
def remove_relacs_files():
    remove_files(relacs_path)

    
def remove_fishgrid_files():
    remove_files(fishgrid_path)

    
def test_container():
    tolerance = 2.0**(-15)
    data, samplerate = generate_data()
    info = dict(Recording=dict(Comment='good', Experimenter='John',
                               Temperature='23.8°C'),
                Subject=dict(Species='Apteronotus leptorhynchus',
                             Sex='Female', Size='12cm'))
    # pickle:
    filename = dw.write_pickle('test', data, samplerate, 'mV', info)
    full_data, rate, unit = dl.load_data(filename, -1)
    assert_true(np.all(np.abs(data - full_data)<tolerance), 'full pickle load failed')
    os.remove(filename)
    filename = dw.write_data('test', data, samplerate, 'mV', format='pickle')
    full_data, rate, unit = dl.load_data(filename, -1)
    assert_true(np.all(np.abs(data - full_data)<tolerance), 'full pickle load failed')
    os.remove(filename)

    # numpy:
    finfo = flatten_metadata(info, True)
    filename = dw.write_numpy('test', data, samplerate, 'mV', finfo)
    full_data, rate, unit = dl.load_data(filename, -1)
    assert_true(np.all(np.abs(data - full_data)<tolerance), 'full numpy load failed')
    os.remove(filename)
    filename = dw.write_data('test', data, samplerate, 'mV', format='numpy')
    full_data, rate, unit = dl.load_data(filename, -1)
    assert_true(np.all(np.abs(data - full_data)<tolerance), 'full pickle load failed')
    os.remove(filename)

    # mat:
    filename = dw.write_mat('test', data, samplerate, 'mV', info)
    full_data, rate, unit = dl.load_data(filename, -1)
    assert_true(np.all(np.abs(data - full_data)<tolerance), 'full mat load failed')
    os.remove(filename)
    filename = dw.write_data('test', data, samplerate, 'mV', format='mat')
    full_data, rate, unit = dl.load_data(filename, -1)
    assert_true(np.all(np.abs(data - full_data)<tolerance), 'full pickle load failed')
    os.remove(filename)
    

def check_reading(filename, data):
    tolerance = 2.0**(-15)

    # load full data:
    full_data, rate, unit = dl.load_data(filename, -1)
    assert_true(np.all(np.abs(data[:-2, :] - full_data)<tolerance), 'full load failed')

    # load on demand:
    data = dl.DataLoader(filename, -1, 10.0, 2.0)

    nframes = int(1.5*data.samplerate)
    # check access:
    ntests = 1000
    step = int(len(data)/ntests)
    success = -1
    print('  check random single frame access...')
    for inx in np.random.randint(0, len(full_data), ntests):
        if np.any(np.abs(full_data[inx] - data[inx]) > tolerance):
            success = inx
            break
    assert_true(success < 0, 'single random frame access failed at index %d' % (success))
    print('  check random frame slice access...')
    for inx in np.random.randint(0, len(full_data)-nframes, ntests):
        if np.any(np.abs(full_data[inx:inx+nframes] - data[inx:inx+nframes]) > tolerance):
            success = inx
            break
    assert_true(success < 0, 'random frame slice access failed at index %d' % (success))
    print('  check forward slice access...')
    for inx in range(0, len(full_data)-nframes, step):
        if np.any(np.abs(full_data[inx:inx+nframes] - data[inx:inx+nframes]) > tolerance):
            success = inx
            break
    assert_true(success < 0, 'frame slice access forward failed at index %d' % (success))
    print('  check backward slice access...')
    for inx in range(len(full_data)-nframes, 0, -step):
        if np.any(np.abs(full_data[inx:inx+nframes] - data[inx:inx+nframes]) > tolerance):
            success = inx
            break
    assert_true(success < 0, 'frame slice access backward failed at index %d' % (success))

    data.close()

@with_setup(None, remove_relacs_files)
def test_relacs():
    data, samplerate = generate_data()
    info = dict(name='foo', date='bar')
    dw.write_relacs(relacs_path, data, samplerate, meta=info)
    dl.relacs_metadata(relacs_path + '/info.dat')
    check_reading(relacs_path, data)


@with_setup(None, remove_fishgrid_files)
def test_fishgrid():
    data, samplerate = generate_data()
    info = dict(name='foo', date='bar')
    dw.write_fishgrid(fishgrid_path, data, samplerate, meta=info)
    check_reading(fishgrid_path, data)

    
def test_main():
    data, samplerate = generate_data()
    filename = dw.write_fishgrid(fishgrid_path, data, samplerate, 'mV')
    dl.main(['-c', '0', filename])
    dl.main(['-p', filename])
    remove_fishgrid_files()
    
