def transcribe(seq):
    '''
    Transcribes a DNA sequence into a vector
    '''
    vocab = {'G' : [1, 0, 0, 0],
             'A' : [0, 1, 0, 0],
             'T' : [0, 0, 1, 0],
             'C' : [0, 0, 0, 1]}
    
    transcription = []
    for c in seq:
        transcription.extend(vocab[c])
    return np.array(transcription)

def patchy(seq, patch_size = 12):
    '''
    Extracts all patches of given size from DNA sequence
    '''
    patches = []
    for i in range(0, len(seq)-patch_size + 1):
        patches.append(transcribe(seq[i:i+patch_size]))
    return patches

def K0(x, xprime):
    '''
    Kernel for Patches
    '''
    k = 12 #patch size
    d = np.linalg.norm(x - xprime)
    sig = 0.1
    return k*np.exp(-d**2/(2*k*sig**2))

def gram_anchors(Z):
    '''
    Computes gram matrix of chosen anchors
    '''
    Kzz = np.zeros((len(Z), len(Z)))
    for i in range(len(Z)):
        for j in range(len(Z)):
            Kzz[i, j] = K0(Z[i], Z[j])
    return Kzz

def project(x, Z):
    '''
    Computes dot products between patch x and the anchors
    '''
    K = []
    for z in Z:
        K.append(K0(x, z))
    return np.array(K)

def embed(seq, Z):
    '''
    Given a list of anchors Z, the function computes a fidi embeding of the DNA sequence
    '''
    Kzz = gram_anchors(Z)
    patches = patchy(seq)
    sqrtK = sqrtm(Kzz)
    fidi = np.zeros(len(Z))
    for patch in patches:
        fidi += np.linalg.solve(sqrtK, project(patch, Z))
    return fidi/len(Z)
        