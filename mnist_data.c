/*
  mnist.c

  Usage:
  $ ./mnist train-images train-labels test-images test-labels
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <endian.h>
#include <string.h>
#include "cnn.h"
#include "mpi.h"

uint32_t be32toh(uint32_t big_endian) {
uint32_t result;
uint8_t *bytes = (uint8_t *)&big_endian;
result = (uint32_t)bytes[0] << 24;
result |= (uint32_t)bytes[1] << 16;
result |= (uint32_t)bytes[2] << 8;
result |= (uint32_t)bytes[3];
return result;
}


/*  IdxFile
 */
typedef struct _IdxFile
{
    int ndims;
    uint32_t* dims;
    uint8_t* data;
} IdxFile;

#define DEBUG_IDXFILE 0

/* IdxFile_read(fp)
   Reads all the data from given fp.
*/
IdxFile* IdxFile_read(FILE* fp)
{
    /* Read the file header. */
    struct {
        uint16_t magic;
        uint8_t type;
        uint8_t ndims;
        /* big endian */
    } header;
    if (fread(&header, sizeof(header), 1, fp) != 1) return NULL;
#if DEBUG_IDXFILE
    fprintf(stderr, "IdxFile_read: magic=%x, type=%x, ndims=%u\n",
            header.magic, header.type, header.ndims);
#endif
    if (header.magic != 0) return NULL;
    if (header.type != 0x08) return NULL;
    if (header.ndims < 1) return NULL;

    /* Read the dimensions. */
    IdxFile* self = (IdxFile*)calloc(1, sizeof(IdxFile));
    if (self == NULL) return NULL;
    self->ndims = header.ndims;
    self->dims = (uint32_t*)calloc(self->ndims, sizeof(uint32_t));
    if (self->dims == NULL) return NULL;
    
    if (fread(self->dims, sizeof(uint32_t), self->ndims, fp) == self->ndims) {
        uint32_t nbytes = sizeof(uint8_t);
        for (int i = 0; i < self->ndims; i++) {
            /* Fix the byte order. */
            uint32_t size = be32toh(self->dims[i]);
#if DEBUG_IDXFILE
            fprintf(stderr, "IdxFile_read: size[%d]=%u\n", i, size);
#endif
            nbytes *= size;
            self->dims[i] = size;
        }
        /* Read the data. */
        self->data = (uint8_t*) malloc(nbytes);
        if (self->data != NULL) {
            fread(self->data, sizeof(uint8_t), nbytes, fp);
#if DEBUG_IDXFILE
            fprintf(stderr, "IdxFile_read: read: %lu bytes\n", n);
#endif
        }
    }

    return self;
}

/* IdxFile_destroy(self)
   Release the memory.
*/
void IdxFile_destroy(IdxFile* self)
{
    assert (self != NULL);
    if (self->dims != NULL) {
        free(self->dims);
        self->dims = NULL;
    }
    if (self->data != NULL) {
        free(self->data);
        self->data = NULL;
    }
    free(self);
}

/* IdxFile_get1(self, i)
   Get the i-th record of the Idx1 file. (uint8_t)
 */
uint8_t IdxFile_get1(IdxFile* self, int i)
{
    assert (self != NULL);
    assert (self->ndims == 1);
    assert (i < self->dims[0]);
    return self->data[i];
}

/* IdxFile_get3(self, i, out)
   Get the i-th record of the Idx3 file. (matrix of uint8_t)
 */
void IdxFile_get3(IdxFile* self, int i, uint8_t* out)
{
    assert (self != NULL);
    assert (self->ndims == 3);
    assert (i < self->dims[0]);
    size_t n = self->dims[1] * self->dims[2];
    memcpy(out, &self->data[i*n], n);
}

int write_weights_biases(Layer* linput, Layer* lconv1, Layer* lconv2, Layer* lfull1, Layer* lfull2, Layer* loutput)
{
    FILE* linputf = fopen("linputf.txt", "r");
    if (linputf == NULL) {
        printf("Error opening file for writing.\n");
        return 1;
    }

    FILE* lconv1f = fopen("lconv1f.txt", "r");
    if (lconv1f == NULL) {
        printf("Error opening file for writing.\n");
        return 1;
    }

    FILE* lconv2f = fopen("lconv2f.txt", "r");
    if (lconv2f == NULL) {
        printf("Error opening file for writing.\n");
        return 1;
    }

    FILE* lfull1f = fopen("lfull1f.txt", "r");
    if (lfull1f == NULL) {
        printf("Error opening file for writing.\n");
        return 1;
    }

    FILE* lfull2f = fopen("lfull2f.txt", "r");
    if (lfull2f == NULL) {
        printf("Error opening file for writing.\n");
        return 1;
    }

    FILE* loutputf = fopen("loutputf.txt", "r");
    if (loutputf == NULL) {
        printf("Error opening file for writing.\n");
        return 1;
    }

    // print layer info in csv
    Layer_details(linput, linputf);
    Layer_details(lconv1, lconv1f);
    Layer_details(lconv2, lconv2f);
    Layer_details(lfull1, lfull1f);
    Layer_details(lfull2, lfull2f);
    Layer_details(loutput, loutputf);

    // Close file
    fclose(linputf);
    fclose(lconv1f);
    fclose(lconv2f);
    fclose(lfull1f);
    fclose(lfull2f);
    fclose(loutputf);

    return 1;
}


int main(int argc, char *argv[]) {
    argv[1] = "./data/train-images-idx3-ubyte";
    argv[2] = "./data/train-labels-idx1-ubyte";
    argv[3] = "./data/t10k-images-idx3-ubyte";
    argv[4] = "./data/t10k-labels-idx1-ubyte";
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    //if (argc < 4) {
        //if (rank == 0) {
    //        fprintf(stderr, "Usage: %s <train-images-file> <train-labels-file> <test-images-file> <test-labels-file>\n", argv[0]);
    //    }
    //    MPI_Finalize();
    //    return 1;
   // }
    double start_time, end_time;

    start_time = MPI_Wtime();

    // Initialize neural network layers
    Layer* linput = Layer_create_input(1, 28, 28);
    Layer* lconv1 = Layer_create_conv(linput, 16, 14, 14, 3, 1, 2, 0.1);
    Layer* lconv2 = Layer_create_conv(lconv1, 32, 7, 7, 3, 1, 2, 0.1);
    Layer* lfull1 = Layer_create_full(lconv2, 200, 0.1);
    Layer* lfull2 = Layer_create_full(lfull1, 200, 0.1);
    Layer* loutput = Layer_create_full(lfull2, 10, 0.1);

    // Load weights for each layer on each process
    FILE* files[] = {
        fopen("linputf.txt", "r"),
        fopen("lconv1f.txt", "r"),
        fopen("lconv2f.txt", "r"),
        fopen("lfull1f.txt", "r"),
        fopen("lfull2f.txt", "r"),
        fopen("loutputf.txt", "r")
    };

    for (int i = 0; i < 6; i++) {
        if (files[i] == NULL) {
            fprintf(stderr, "Error opening file for reading.\n");
            MPI_Finalize();
            return 1;
        }
    }

    Load_pretrainedValues(linput, files[0]);
    Load_pretrainedValues(lconv1, files[1]);
    Load_pretrainedValues(lconv2, files[2]);
    Load_pretrainedValues(lfull1, files[3]);
    Load_pretrainedValues(lfull2, files[4]);
    Load_pretrainedValues(loutput, files[5]);

    // Close files
    for (int i = 0; i < 6; i++) {
        fclose(files[i]);
    }

    // Image processing
    IdxFile* images_test = NULL;
    {
        FILE* fp = fopen(argv[3], "rb");
        if (fp == NULL) {
            MPI_Finalize();
            return 111;
        }
        images_test = IdxFile_read(fp);
        if (images_test == NULL) {
            MPI_Finalize();
            return 111;
        }
        fclose(fp);
    }
    IdxFile* labels_test = NULL;
    {
        FILE* fp = fopen(argv[4], "rb");
        if (fp == NULL) {
            MPI_Finalize();
            return 111;
        }
        labels_test = IdxFile_read(fp);
        if (labels_test == NULL) {
            MPI_Finalize();
            return 111;
        }
        fclose(fp);
    }

    // fprintf(stderr, "testing...\n");
    int ntests = images_test->dims[0];
    int ncorrect = 0;
    //printf("no error at ntest..\n");
    // Calculate how many images each process will handle
    int images_per_process = ntests / size;
    int remainder = ntests % size;

    int start_index = rank * images_per_process;
    int end_index = start_index + images_per_process;
    //printf("no error at image indexing..\n");
    if (rank == size - 1) {
        // Last process takes care of any remaining images
        end_index += remainder;
    }
   //printf("no error at image indexing for last processor..\n");
    for (int i = start_index; i < end_index; i++) {
        uint8_t img[28*28];
        double x[28*28];
        double y[10];
        IdxFile_get3(images_test, i, img);
        for (int j = 0; j < 28*28; j++) {
            x[j] = img[j] / 255.0;
        }
        //printf("no error reading images..\n");
	// Process the image...
        Layer_setInputs(linput, x);
	//printf("no error inserting inputs..\n");
        Layer_getOutputs(loutput, y);
        //printf("no error processing images..\n");
        // Read label
        int label = IdxFile_get1(labels_test, i);
        
        /* Pick the most probable label. */
        int mj = -1;
        for (int j = 0; j < 10; j++) {
            if (mj < 0 || y[mj] < y[j]) {
                mj = j;
            }
        }
        if (mj == label) {
            ncorrect++;
        }
        
        if ((i % 1000) == 0) {
            fprintf(stderr, "i=%d\n", i);
        }
    }

    // Reduce ncorrect across all processes
    int total_correct;
    MPI_Reduce(&ncorrect, &total_correct, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Total correct predictions: %d\n", total_correct);
    }
    end_time = MPI_Wtime();
    double execution_time = end_time - start_time;
    if(rank ==0)
	printf("Total execution time: %f seconds\n", execution_time);

    MPI_Finalize();
    return 0;
}

