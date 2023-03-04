#!/usr/bin/env python
# Copyright 2014-2016  Brno University of Technology (author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

import gzip
import logging
import os
import re
import struct

import numpy as np

logger = logging.getLogger(__name__)

#################################################
# Data-type independent helper functions,


def open_or_fd(file, mode="rb"):
    """fd = open_or_fd(file)
    Open file, gzipped file, pipe, or forward the file-descriptor.
    Eventually seeks in the 'file' argument contains ':offset' suffix.
    Parameters
    ----------
    file : obj
        An ark, gzipped ark, pipe or opened file descriptor.
    mode : str, optional
        A mode of the file.
    Returns
    -------
    obj
        A file descriptor.
    """
    offset = None
    try:
        # strip 'ark:' prefix from r{x,w}filename (optional),
        if re.search("^(ark|scp)(,scp|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:", file):
            (prefix, file) = file.split(":", 1)
        # separate offset from filename (optional),
        if re.search(":[0-9]+$", file):
            (file, offset) = file.rsplit(":", 1)
        # is it gzipped?
        if file.split(".")[-1] == "gz":
            fd = gzip.open(file, mode)
        # input pipe?
        elif file[-1] == "|":
            fd = os.popen(file[:-1], "rb")
        # output pipe?
        elif file[0] == "|":
            fd = os.popen(file[1:], "wb")
        # a normal file...
        else:
            fd = open(file, mode)
    except TypeError:
        # 'file' is opened file descriptor,
        fd = file
    # Eventually seek to offset,
    if offset is not None:
        fd.seek(int(offset))
    return fd


def read_key(fd):
    """[str] = read_key(fd)
    Read the utterance-key from the opened ark/stream descriptor 'fd'.
    Parameters
    ----------
    fd : obj
        An ark, gzipped ark, pipe or opened file descriptor.
    Returns
    -------
    str
        An utterance-key.
    """
    str = b""
    while 1:
        char = fd.read(1)
        if char == b"":
            break
        if char == b" ":
            break
        str += char
    str = str.strip()
    if str == b"":
        return None  # end of file,
    assert re.match(b"^[\\.a-zA-Z0-9_<>-]+$", str) is not None  # check format,
    return str


#################################################
# Integer vectors (alignments, ...),


def read_ali_ark(file_or_fd):
    """ Alias to 'read_vec_int_ark()' """
    return read_vec_int_ark(file_or_fd)


def read_vec_int_ark(file_or_fd):
    """generator(key,vec) = read_vec_int_ark(file_or_fd)
    Create generator of (key,vector<int>) tuples, which reads from the ark
    file/stream.
    Read ark to a 'dictionary':
    d = { u:d for u,d in kaldi_io.read_vec_int_ark(file) }
    Parameters
    ----------
    file_or_fd : obj
        An ark, gzipped ark, pipe or opened file descriptor.
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            ali = read_vec_int(fd)
            yield key, ali
            key = read_key(fd)
    finally:
        if fd is not file_or_fd:
            fd.close()


def read_vec_int(file_or_fd):
    """[int-vec] = read_vec_int(file_or_fd)
    Read kaldi integer vector, ascii or binary input,
    Parameters
    ----------
    file_or_fd : obj
        An ark, gzipped ark, pipe or opened file descriptor.
    Returns
    -------
    list of int
        Returns integer vector.
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2)
    if binary == "\0B":  # binary flag
        assert fd.read(1) == "\4"  # int-size
        vec_size = struct.unpack("<i", fd.read(4))[0]  # vector dim
        ans = np.zeros(vec_size, dtype=int)
        for i in range(vec_size):
            assert fd.read(1) == "\4"  # int-size
            ans[i] = struct.unpack("<i", fd.read(4))[0]  # data
        return ans
    else:  # ascii,
        arr = (binary + fd.readline()).strip().split()
        try:
            arr.remove("[")
            arr.remove("]")  # optionally
        except ValueError:
            pass
        ans = np.array(arr, dtype=int)
    if fd is not file_or_fd:
        fd.close()  # cleanup
    return ans


# Writing,


def write_vec_int(file_or_fd, v, key=""):
    """write_vec_int(f, v, key='')
    Write a binary kaldi integer vector to filename or stream.
    Parameters
    ----------
    file_or_fd: obj
        filename or opened file descriptor for writing,
    v: list of int
        the vector to be stored,
    key : str, optional
        used for writing ark-file, the utterance-id gets written
        before the vector.
    Example of writing single vector
    --------------------------------
    kaldi_io.write_vec_int(filename, vec)
    Example of writing arkfile
    --------------------------
    with open(ark_file,'w') as f:
        for key,vec in dict.iteritems():
            kaldi_io.write_vec_flt(f, vec, key=key)
    """
    fd = open_or_fd(file_or_fd, mode="wb")
    try:
        keyd = key + " "
        if key != "":
            # ark-files have keys (utterance-id),
            fd.write(keyd.encode("utf-8"))
        fd.write(b"\0B")  # we write binary!
        # dim,
        fd.write("\4")  # int32 type,
        fd.write(struct.pack("<i", v.shape[0]))
        # data,
        for i in range(len(v)):
            fd.write("\4")  # int32 type,
            fd.write(struct.pack("<i", v[i]))  # binary,
    finally:
        if fd is not file_or_fd:
            fd.close()


#################################################
# Float vectors (confidences, ivectors, ...),


def read_vec_flt_ark(file_or_fd):
    """generator(key,vec) = read_vec_flt_ark(file_or_fd)
    Create generator of (key,vector<float>) tuples, reading from an ark
    file/stream.
    Read ark to a 'dictionary':
    d = { u:d for u,d in kaldi_io.read_vec_flt_ark(file) }
    Parameters
    ----------
    file_or_fd : obj
        An ark, gzipped ark, pipe or opened file descriptor.
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            ali = read_vec_flt(fd)
            yield key, ali
            key = read_key(fd)
    finally:
        if fd is not file_or_fd:
            fd.close()


def read_vec_flt(file_or_fd):
    """[flt-vec] = read_vec_flt(file_or_fd)
    Read kaldi float vector, ascii or binary input,
    Parameters
    ----------
    file_or_fd : obj
        An ark, gzipped ark, pipe or opened file descriptor.
    Raises
    ------
    ValueError
        Unsupported data-type of the input file.
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2)
    if binary == b"\0B":  # binary flag
        # Data type,
        type = fd.read(3)
        if type == b"FV ":
            sample_size = 4  # floats
        if type == b"DV ":
            sample_size = 8  # doubles
        assert sample_size > 0
        # Dimension,
        assert fd.read(1) == b"\4"  # int-size
        vec_size = struct.unpack("<i", fd.read(4))[0]  # vector dim
        # Read whole vector,
        buf = fd.read(vec_size * sample_size)
        if sample_size == 4:
            ans = np.frombuffer(buf, dtype="float32")
        elif sample_size == 8:
            ans = np.frombuffer(buf, dtype="float64")
        else:
            raise ValueError("BadSampleSize")
        return ans
    else:  # ascii,
        arr = (binary + fd.readline()).strip().split()
        try:
            arr.remove("[")
            arr.remove("]")  # optionally
        except ValueError:
            pass
        ans = np.array(arr, dtype=float)
    if fd is not file_or_fd:
        fd.close()  # cleanup
    return ans


# Writing,


def write_vec_flt(file_or_fd, v, key=b""):
    """write_vec_flt(f, v, key='')
    Write a binary kaldi vector to filename or stream. Supports 32bit and
    64bit floats.
    Parameters
    ----------
    file_or_fd: obj
        An ark, gzipped ark, pipe or opened file descriptor.
    v: str
        the vector to be stored,
    key : str, optional
        used for writing ark-file, the utterance-id gets written
        before the vector.
    Example of writing single vector
    --------------------------------
    kaldi_io.write_vec_flt(filename, vec)
    Example of writing arkfile
    --------------------------
    with open(ark_file,'w') as f:
        for key,vec in dict.iteritems():
            kaldi_io.write_vec_flt(f, vec, key=key)
    Raises
    ------
    ValueError
        Unsupported data-type of the input file.
    """
    fd = open_or_fd(file_or_fd, mode="wb")
    try:
        if key != b"":
            fd.write(key + b" ")  # ark-files have keys (utterance-id),
        fd.write(b"\0B")  # we write binary!
        # Data-type,
        if v.dtype == "float32":
            fd.write(b"FV ")
        elif v.dtype == "float64":
            fd.write(b"DV ")
        else:
            raise ValueError("VectorDataTypeError")
        # Dim,
        fd.write(b"\04")
        fd.write(struct.pack("I", v.shape[0]))  # dim
        # Data,
        # v.tofile(fd, sep="") # binary
        # v.tofile(fd) # binary by default
        fd.write(v.tobytes())
    finally:
        if fd is not file_or_fd:
            fd.close()


#################################################
# Float matrices (features, transformations, ...),

# Reading,
def read_mat_scp(file_or_fd):
    """generator(key,mat) = read_mat_scp(file_or_fd)
    Returns generator of (key,matrix) tuples, read according to kaldi scp.
    file_or_fd : scp, gzipped scp, pipe or opened file descriptor.
    Iterate the scp
    ---------------
    for key,mat in kaldi_io.read_mat_scp(file):
        ...
    Read scp to a 'dictionary':
    d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }
    Parameters
    ----------
    file_or_fd : obj
        An ark, gzipped ark, pipe or opened file descriptor.
    """
    fd = open_or_fd(file_or_fd)
    try:
        for line in fd:
            (key, rxfile) = line.split(" ")
            mat = read_mat(rxfile)
            yield key, mat
    finally:
        if fd is not file_or_fd:
            fd.close()


def read_plda(file_or_fd):
    """res = read_plda(file_or_fd)
    Returns a dictionary {'mean':plda_mean, 'transform':plda_transform, 'psi':plda_psi}, read from ark file/stream.
    file_or_fd : scp, gzipped scp, pipe or opened file descriptor.
    Parameters
    ----------
    file_or_fd : obj
        An ark, gzipped ark, pipe or opened file descriptor.
    """
    fd = open_or_fd(file_or_fd)
    try:
        assert fd.read(2) == b"\0B"
        plda_tag = read_key(fd)
        assert plda_tag == b"<Plda>"

        mean = read_mat(fd, True)
        transform = read_mat(fd, True)
        psi = read_mat(fd, True)

        return {"mean": mean, "transform": transform, "psi": psi}

    finally:
        if fd is not file_or_fd:
            fd.close()


def read_mat_ark(file_or_fd, read_binary=False):
    """generator(key,mat) = read_mat_ark(file_or_fd)
    Returns generator of (key,matrix) tuples, read from ark file/stream.
    file_or_fd : scp, gzipped scp, pipe or opened file descriptor.
    Iterate the ark
    ---------------
    for key,mat in kaldi_io.read_mat_ark(file):
        ...
    Read ark to a 'dictionary':
    d = { key:mat for key,mat in kaldi_io.read_mat_ark(file) }
    Parameters
    ----------
    file_or_fd : obj
        An ark, gzipped ark, pipe or opened file descriptor.
    read_binary : boolean
        If set directly read binary mat
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            mat = read_mat(fd, read_binary)
            yield key, mat
            key = read_key(fd)
    finally:
        if fd is not file_or_fd:
            fd.close()


def read_mat(file_or_fd, read_binary=False):
    """[mat] = read_mat(file_or_fd)
    Reads single kaldi matrix, supports ascii and binary.
    Parameters
    ----------
    file_or_fd : obj
        An ark, gzipped ark, pipe or opened file descriptor.
    read_binary : boolean
        If set directly read binary mat
    """
    fd = open_or_fd(file_or_fd)
    try:
        if not read_binary:
            binary = fd.read(2)
            if binary == b"\0B":
                mat = _read_mat_binary(fd)
            else:
                assert binary == b" ["
                mat = _read_mat_ascii(fd)
        else:
            mat = _read_mat_binary(fd)
    finally:
        if fd is not file_or_fd:
            fd.close()
    return mat


def _read_mat_binary(fd):
    # Data type
    type = fd.read(3)
    if type == b"FM " or type == b"FV ":
        sample_size = 4  # floats
    elif type == b"DM " or type == b"DV ":
        sample_size = 8  # doubles

    read_vector = False
    if type == b"FV " or type == b"DV ":
        read_vector = True

    assert sample_size > 0
    # Dimensions
    fd.read(1)
    rows = struct.unpack("<i", fd.read(4))[0]
    if not read_vector:
        fd.read(1)
        cols = struct.unpack("<i", fd.read(4))[0]
    else:
        cols = 1
    # Read whole matrix
    buf = fd.read(rows * cols * sample_size)
    if sample_size == 4:
        vec = np.frombuffer(buf, dtype="float32")
    elif sample_size == 8:
        vec = np.frombuffer(buf, dtype="float64")
    else:
        raise ValueError("BadSampleSize")
    mat = np.reshape(vec, (rows, cols))
    return mat


def _read_mat_ascii(fd):
    rows = []
    while 1:
        line = fd.readline()
        if len(line) == 0:
            raise ValueError("BadInputFormat")  # eof, should not happen!
        if len(line.strip()) == 0:
            continue  # skip empty line
        arr = line.strip().split()
        if arr[-1] != b"]":
            rows.append(np.array(arr, dtype="float32"))  # not last line
        else:
            rows.append(np.array(arr[:-1], dtype="float32"))  # last line
            mat = np.vstack(rows)
            return mat


# Writing,


def write_vec(file_or_fd, v, key=b""):
    """write_vec(f, v, key='')
    Write a binary kaldi vector to filename or stream. Supports 32bit and 64bit
    floats.
    Parameters
    ----------
    file_or_fd: obj
        filename of opened file descriptor for writing,
    v: numpy.ndarray
        the vector or matrix to be stored, for matrix it will be reshaped to vector in C order,
    key : str, optional
        used for writing ark-file, the utterance-id gets written before the
        vector.
    Example of writing single vector:
    kaldi_io.write_vec(filename, vec)
    Example of writing arkfile:
    with open(ark_file,'w') as f:
        for key,vec in dict.iteritems():
            kaldi_io.write_vec(f, vec, key=key)
    Raises
    ------
    ValueError
        Unsupported data-type of the input file.
    """
    fd = open_or_fd(file_or_fd, mode="wb")
    try:
        if key != b"":
            fd.write(key + b" ")  # ark-files have keys (utterance-id),
        fd.write(b"\0B")  # we write binary!
        # Data-type,
        if v.dtype == "float32":
            fd.write(b"FV ")
        elif v.dtype == "float64":
            fd.write(b"DV ")
        else:
            raise ValueError("MatrixDataTypeError")
        # Dims,
        dim = v.size
        fd.write(b"\04")
        fd.write(struct.pack("I", dim))  # vector length
        # Data,
        # m.tofile(fd, sep=b"") # binary
        fd.write(v.tobytes())

    finally:
        if fd is not file_or_fd:
            fd.close()


def write_mat(file_or_fd, m, key=b""):
    """write_mat(f, m, key='')
    Write a binary kaldi matrix to filename or stream. Supports 32bit and 64bit
    floats.
    Parameters
    ----------
    file_or_fd: obj
        filename of opened file descriptor for writing,
    m: numpy.ndarray
        the matrix to be stored,
    key : str, optional
        used for writing ark-file, the utterance-id gets written before the
        matrix.
    Example of writing single matrix:
    kaldi_io.write_mat(filename, mat)
    Example of writing arkfile:
    with open(ark_file,'w') as f:
        for key,mat in dict.iteritems():
            kaldi_io.write_mat(f, mat, key=key)
    Raises
    ------
    ValueError
        Unsupported data-type of the input file.
    """
    fd = open_or_fd(file_or_fd, mode="wb")
    try:
        if key != b"":
            fd.write(key + b" ")  # ark-files have keys (utterance-id),
        fd.write(b"\0B")  # we write binary!
        # Data-type,
        if m.dtype == "float32":
            fd.write(b"FM ")
        elif m.dtype == "float64":
            fd.write(b"DM ")
        else:
            raise ValueError("MatrixDataTypeError")
        # Dims,
        fd.write(b"\04")
        fd.write(struct.pack("I", m.shape[0]))  # rows
        fd.write(b"\04")
        fd.write(struct.pack("I", m.shape[1]))  # cols
        # Data,
        # m.tofile(fd, sep=b"") # binary
        fd.write(m.tobytes())

    finally:
        if fd is not file_or_fd:
            fd.close()


#################################################
# 'Posterior' kaldi type (posteriors, confusion network, nnet1 training targets, ...)
# Corresponds to: vector<vector<tuple<int,float> > >
# - outer vector: time axis
# - inner vector: records at the time
# - tuple: int = index, float = value
#


def read_cnet_ark(file_or_fd):
    """Alias of function 'read_post_ark()', 'cnet' = confusion network
    Parameters
    ----------
    file_or_fd : obj
        An ark, gzipped ark, pipe or opened file descriptor.
    """
    return read_post_ark(file_or_fd)


def read_post_ark(file_or_fd):
    """generator(key,vec<vec<int,float>>) = read_post_ark(file)
    Returns generator of (key,posterior) tuples, read from ark file.
    file_or_fd : ark, gzipped ark, pipe or opened file descriptor.
    Iterate the ark
    ---------------
    for key,post in kaldi_io.read_post_ark(file):
        ...
    Read ark to a 'dictionary':
    d = { key:post for key,post in kaldi_io.read_post_ark(file) }
    Parameters
    ----------
    file_or_fd : obj
        An ark, gzipped ark, pipe or opened file descriptor.
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            post = read_post(fd)
            yield key, post
            key = read_key(fd)
    finally:
        if fd is not file_or_fd:
            fd.close()


def read_post(file_or_fd):
    """[post] = read_post(file_or_fd)
    Reads single kaldi 'Posterior' in binary format.
    The 'Posterior' is C++ type 'vector<vector<tuple<int,float> > >',
    the outer-vector is usually time axis, inner-vector are the records
    at given time,  and the tuple is composed of an 'index' (integer)
    and a 'float-value'. The 'float-value' can represent a probability
    or any other numeric value.
    Returns vector of vectors of tuples.
    Parameters
    ----------
    file_or_fd : obj
        An ark, gzipped ark, pipe or opened file descriptor.
    """
    fd = open_or_fd(file_or_fd)
    ans = []
    binary = fd.read(2)
    assert binary == b"\0B"  # binary flag
    assert fd.read(1) == b"\4"  # int-size
    outer_vec_size = struct.unpack("<i", fd.read(4))[0]  # number of frames (or bins)

    # Loop over 'outer-vector',
    for i in range(outer_vec_size):
        assert fd.read(1) == b"\4"  # int-size
        # number of records for frame (or bin)
        inner_vec_size = struct.unpack("<i", fd.read(4))[0]
        id = np.zeros(inner_vec_size, dtype=int)  # buffer for integer id's
        post = np.zeros(inner_vec_size, dtype=float)  # buffer for posteriors

        # Loop over 'inner-vector',
        for j in range(inner_vec_size):
            assert fd.read(1) == b"\4"  # int-size
            id[j] = struct.unpack("<i", fd.read(4))[0]  # id
            assert fd.read(1) == b"\4"  # float-size
            post[j] = struct.unpack("<f", fd.read(4))[0]  # post

        # Append the 'inner-vector' of tuples into the 'outer-vector'
        ans.append(zip(id, post))

    if fd is not file_or_fd:
        fd.close()
    return ans


#################################################
# Kaldi Confusion Network bin begin/end times,
# (kaldi stores CNs time info separately from the Posterior).
#


def read_cntime_ark(file_or_fd):
    """generator(key,vec<tuple<float,float>>) = read_cntime_ark(file_or_fd)
    Returns generator of (key,cntime) tuples, read from ark file.
    file_or_fd : file, gzipped file, pipe or opened file descriptor.
    Iterate the ark
    ---------------
    for key,time in kaldi_io.read_cntime_ark(file):
        ...
    Read ark to a 'dictionary':
    d = { key:time for key,time in kaldi_io.read_post_ark(file) }
    Parameters
    ----------
    file_or_fd : obj
        An ark, gzipped ark, pipe or opened file descriptor.
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            cntime = read_cntime(fd)
            yield key, cntime
            key = read_key(fd)
    finally:
        if fd is not file_or_fd:
            fd.close()


def read_cntime(file_or_fd):
    """[cntime] = read_cntime(file_or_fd)
    Reads single kaldi 'Confusion Network time info', in binary format:
    C++ type: vector<tuple<float,float> >.
    (begin/end times of bins at the confusion network).
    Binary layout is '<num-bins> <beg1> <end1> <beg2> <end2> ...'
    file_or_fd : file, gzipped file, pipe or opened file descriptor.
    Returns vector of tuples.
    Parameters
    ----------
    file_or_fd : obj
        An ark, gzipped ark, pipe or opened file descriptor.
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2)
    assert binary == b"\0B"  # assuming it's binary
    assert fd.read(1) == b"\4"  # int-size
    # Get number of bins,
    vec_size = struct.unpack("<i", fd.read(4))[0]  # number of frames (or bins)
    t_beg = np.zeros(vec_size, dtype=float)
    t_end = np.zeros(vec_size, dtype=float)

    # Loop over number of bins,
    for i in range(vec_size):
        assert fd.read(1) == b"\4"  # float-size
        t_beg[i] = struct.unpack("<f", fd.read(4))[0]  # begin-time of bin
        assert fd.read(1) == b"\4"  # float-size
        t_end[i] = struct.unpack("<f", fd.read(4))[0]  # end-time of bin

    # Return vector of tuples,
    ans = zip(t_beg, t_end)
    if fd is not file_or_fd:
        fd.close()
    return ans


def write_wav(fp, data, rate, bitdepth=16):
    """ Write a wav file.
    Parameters
    ----------
    fp : obj
        A file pointer.
    data: numpy.ndarray
        Data samples
    rate: int
        Sample rate.
    bitdepth: int, optional
        A bit depth of samples.
    """
    num_chan = 1
    num_samp = data.size

    # logger.debug("... write_wav. num_samp: %s", num_samp)

    assert bitdepth == 8 or bitdepth == 16 or bitdepth == 32

    if bitdepth == 8:
        bytes_per_samp = 1
    elif bitdepth == 16:
        bytes_per_samp = 2
    elif bitdepth == 32:
        bytes_per_samp = 4

    subchunk2size = num_chan * num_samp * bytes_per_samp
    chunk_size = 36 + subchunk2size

    fp.write(b"RIFF")
    fp.write(np.array(chunk_size, dtype="int32"))
    fp.write(b"WAVE")
    fp.write(b"fmt ")
    fp.write(np.array(16, dtype="int32"))
    # wFormatTag, 1 for PCM, 3 for IEEE_FLOAT
    # Kaldi will only read PCM wav files
    fp.write(np.array(1, dtype="int16"))
    fp.write(np.array(num_chan, dtype="int16"))
    # logger.debug("... write_wav",)

    fp.write(np.array(rate, dtype="int32"))
    fp.write(np.array(rate * num_chan * bytes_per_samp, dtype="int32"))
    fp.write(np.array(num_chan * bytes_per_samp, dtype="int16"))
    fp.write(np.array(8 * bytes_per_samp, dtype="int16"))

    fp.write(b"data")
    fp.write(np.array(subchunk2size, dtype="int32"))

    # import pdb; pdb.set_trace()
    if bitdepth == 8:
        scale = 1 << 7
        fp.write(np.asarray(scale * data, dtype="int8"))
    elif bitdepth == 16:
        scale = 1 << 15
        fp.write(np.asarray(scale * data, dtype="int16"))
    elif bitdepth == 32:
        scale = 1 << 31
        fp.write(np.asarray(scale * data, dtype="uint32"))

    return