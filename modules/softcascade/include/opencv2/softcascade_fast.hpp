/*
 * softcascade_plus.hpp
 *
 *  Created on: Oct 29, 2013
 *      Author: federico
 */

#ifndef SOFTCASCADE_FAST_HPP_
#define SOFTCASCADE_FAST_HPP_

#include "softcascade.hpp"
#include <fstream>
#include <map>
#include <list>
#include <numeric>
#include <set>
#include <iostream>

struct Level;
struct ChannelStorage;

/*

#define DEBUG_MSG

#ifdef DEBUG_MSG
#define DEBUG_MSG(str) do { std::cout << "<<debug>>" << str << std::endl; } while( false )
#else
#define DEBUG_MSG(str) do { } while ( false )
#endif
*/

namespace cv { namespace softcascade {

// ============================================================================================================== //
//					     Declaration of DetectorFast (with trace evaluation reduction)
// ============================================================================================================= //
struct CV_EXPORTS ParamDetectorFast
{

	ParamDetectorFast();
	ParamDetectorFast(double minScale, double maxScale, uint nScale, int nMS, uint lastStage, uint gridSize,double gamma,uint round,bool exp, bool unE);

	// pyramid settings
	double	minScale;
	double	maxScale;
	uint 	nScales;

	// No Maximum Soppression
	int 	nMS;

	// Linear Cascade Approximation parameter
	uint 	lastStage;

	// Geometric model: grid size
	int		geomModelType;
	uint 	gridSize;
	double 	gamma;
	uint 	round;
	bool 	covMExpansion;
	bool	uniformEnergy;

};


struct CV_EXPORTS FastDtModel
{
	// tag for model (xml file)
	static const char *const ROOT;
	static const char *const PYRAMID;
	static const char *const PYRAMID_MINS;
	static const char *const PYRAMID_MAXS;
	static const char *const PYRAMID_NS;

	static const char *const TRAININGS;
	static const char *const TRAININGS_DATAF;
	static const char *const TRAININGS_NIMG;
	static const char *const TRAININGS_IMGS;

	static const char *const MODELS;

	struct Block;
	struct MixtureC;

	FastDtModel(ParamDetectorFast paramDtFast, std::string dataset,uint numImages,Size imgSize);
	FastDtModel();
	//FastDtModel(uint numLevels);


    void write(cv::FileStorage& fso) const;
    void read(const cv::FileNode& node);

    // Interface for Trace-Model
    void addTraceForTraceModel(uint stage,uint level,const std::vector<Point2d>& trace);
    void computeModel(bool useTraceApp,bool useGeomHist);
    bool getSlopeAt(uint stage,uint level,double& slope);
    void getLastSt(std::vector<uint>& stages);
    bool getLevelsForStage(uint  lastStage, std::vector<uint>& levels);

    // Interface for Geometry-Model
    void addStrongWithROI(Rect dw,double rank,uint level);
    void setGridsSize(std::vector<uint> grids);
    std::vector<Block>& getBlocks4Grid(uint gridSize);
    MixtureC& getMixtureCAtLevel(uint level){return geomModelGMM.gmm[level];};

    void resolveWrongStd();
    void smoothLocations();
    void saveModelIntoDat(String path,String prefix="");


    static double det(const Mat& inM){
    	return inM.at<double>(0,0)*inM.at<double>(1,1) - (inM.at<double>(0,1)*inM.at<double>(1,0));
    }

    // for now, merging geometric models only
    static FastDtModel mergeModels(const std::vector<FastDtModel>& models){
    	FastDtModel outModel;

    		if (models.size()==0)
    			return outModel;
    		if (models.size()==1)
    			return models[0];

    		outModel=models[0];

    		// Trace Approximation
    		// empty

    		// Geometry Model
    		GeomModel::Grids::iterator itG=outModel.geomModel.grids.begin();
    		for( ;itG!=outModel.geomModel.grids.end();++itG){

    			// For each block
    			double energyAcc=0.;
    			for(std::vector<Block>::iterator itB=itG->second.begin();itB!=itG->second.end();++itB){
    				double energyAvg=itB->energy;
    				for(uint l=0;l<itB->levelsHist.size();l++){
    					itB->levelsHist[l]*=itB->energy;

    					// merging levels histogram
    					for(uint m=1;m<models.size();m++){
    						const Block* block=&(models.at(m).geomModel.grids.at(itG->first).at(itB-itG->second.begin()));
    						itB->levelsHist[l]+=block->levelsHist[l]*block->energy;
    						energyAvg+=block->energy;
    					}


    				}
    				// levels histogram normalization
    				double levHistAcc=std::accumulate(itB->levelsHist.begin(),itB->levelsHist.end(),0.);
    				std::transform(itB->levelsHist.begin(), itB->levelsHist.end(), itB->levelsHist.begin(),
    				               std::bind1st(std::multiplies<double>(),1./levHistAcc));

    				// assign new block energy
    				itB->energy=energyAvg;
    				energyAcc+=energyAcc;
    			}

    			// blocks energy normalization
    			for(std::vector<Block>::iterator itB=itG->second.begin();itB!=itG->second.end();++itB)
    				itB->energy/=energyAcc;
    		}
    		return outModel;
    }
    static FastDtModel UNIFORM_MODEL(const ParamDetectorFast& prm,const Size& imgSize, const std::vector<uint>& grids){

    	FastDtModel outModel(prm, "",-1,imgSize);
    	outModel.setGridsSize(grids);

    	// set geom. uniformly
    	outModel.geomModel.setUniform(imgSize,prm.nScales);

    	return outModel;
    }


    // ------------ Parameters ---------------------

    // List of octaves of the soft cascade (in logarithmic scale)
    std::vector<int> octaves;
    // List of levels of the soft cascade
    std::vector<double> levels;

    ParamDetectorFast paramDtFast;

    // Additional information for training of trace/geometry models
    std::string	dataset;
    uint	numImages;
    Size	imgSize;

	struct AverageCov{
		AverageCov(){
			avg=Mat(1,2,CV_64FC1,-1.);
			cov=Mat(2,2,CV_64FC1,0.);
		};
		AverageCov(Mat a, Mat c): avg(a),cov(c){};

		Mat avg;
		Mat cov;
	};

	struct Block{
		Block(uint levels)
		:levelsHist(std::vector<double>(levels,0.)),
		locationsHist(std::vector<AverageCov>(levels,AverageCov())),
		energy(0.){};
		Block(uint levels, double histV, double energyV)
				:levelsHist(std::vector<double>(levels,histV)),
				locationsHist(std::vector<AverageCov>(levels,AverageCov())),
				energy(energyV){};



		Block(std::vector<double> lvH,std::vector<AverageCov> locH, Rect rt, double e)
		:levelsHist(lvH), locationsHist(locH), rect(rt), energy(e){};

		std::vector<double>  	levelsHist;
		std::vector<AverageCov> locationsHist;
		Rect rect;
		double 					energy;
 	};

	struct MixtureC{
		MixtureC(){
			avgCov=std::vector<AverageCov>();
			mixingP=std::vector<double>();
			energy=std::vector<double>();
		};

		std::vector<AverageCov> avgCov;
		std::vector<double> mixingP;
		std::vector<double> energy;

	};



private:
    struct TraceModel{
#define Vx  0
#define Vy  1

    	static const char *const TRACEMODEL;
    	static const char *const TRACEMODEL_LASTSTAGES;
    	static const char *const TRACEMODEL_LASTSTAGES_LASTS;
    	static const char *const TRACEMODEL_LASTSTAGES_SLOPES;



    	typedef std::map<uint,std::map<uint,std::vector<Vec4f> > > LinesMap;
    	typedef std::map<uint,std::vector<Vec4f> >  LevelsMap;
    	//typedef std::map<uint,std::map<uint,double > > SlopesMap;
    	typedef std::map<uint,std::vector<double> > SlopesMap;


    	TraceModel(){}

    	void compute(uint levels);
    	void write(FileStorage& fso) const;
    	void read(const FileNode& node);

    	//output line parameter in affine coordinate
    	//stage->level->lines
    	//std::map<uint,std::map<uint,std::vector<Vec4f> > > linesParam;
    	//std::map<uint,std::map<uint,double> > slopes;
    	LinesMap linesParam;
    	SlopesMap slopes;

    }traceModel;

    struct GeomModel{
    	static const char *const GEOMMODEL;
    	static const char *const GEOMMODEL_GRIDS;
    	static const char *const GEOMMODEL_GRID_SIZE;
    	static const char *const GEOMMODEL_GRID_BLOCKS;
    	static const char *const GEOMMODEL_GRID_BLOCKS_ID;
    	static const char *const GEOMMODEL_GRID_BLOCKS_LEVELSH;
    	static const char *const GEOMMODEL_GRID_BLOCKS_LOCATIONSH;
    	static const char *const GEOMMODEL_GRID_BLOCKS_LOCATIONSH_AVG;
    	static const char *const GEOMMODEL_GRID_BLOCKS_LOCATIONSH_COV;
    	static const char *const GEOMMODEL_GRID_BLOCKS_RECT;
    	static const char *const GEOMMODEL_GRID_BLOCKS_ENERGY;

    	struct StrongROI{
    		StrongROI(Rect d, double r):dw(d),rank(r){};

    		Rect dw;
    		double rank;

    	};

    	typedef std::map<uint,std::vector<StrongROI> > StrongsROI;
    	typedef std::map<uint,std::vector<Block> > Grids;

    	GeomModel(){}

    	void compute(Size imageSize,uint levels);
    	void write(FileStorage& fso) const;
    	void read(const FileNode& node);

    	void setUniform(Size imgSize,uint levels);


    	// variables for storage input data
    	StrongsROI  		upperLeftPonts;
    	std::vector<uint> 	gridsSize;

    	// model
    	Grids	grids;

    }geomModel;

    struct GeomModelGMM{
    	static const char *const GEOMMODELGMM;
    	static const char *const GEOMMODELGMM_LEVELS;
    	static const char *const GEOMMODELGMM_LEVELS_AVG;
    	static const char *const GEOMMODELGMM_LEVELS_COV;
    	static const char *const GEOMMODELGMM_LEVELS_MIXC;
    	static const char *const GEOMMODELGMM_LEVELS_ENERGY;

    	GeomModelGMM(){
    		gmm=std::vector< MixtureC>();

    	}



    	void write(FileStorage& fso) const;
    	void read(const FileNode& node);

    	// variables for storage input data
    	std::vector<MixtureC> gmm;
    }geomModelGMM;
};



// required for cv::FileStorage serialization
inline void write(cv::FileStorage& fso, const std::string&, const FastDtModel& x){
	x.write(fso);
}
inline void read(const cv::FileNode& node, FastDtModel& x, const FastDtModel& default_value){
	if(node.empty())
		x=default_value;
	else
		x.read(node);

}
// For print FastModel to the console
//std::ostream& operator<<(std::ostream& out, const FastDtModel& m);


struct classPoint2iComp{
	inline bool operator()(const Point2i& a, const Point2i& b){
		return (a.x<=b.x)&&(a.y!=b.y);

	}
};
struct classPoint3iComp{
	inline bool operator()(const Point3i& a, const Point3i& b){
		return (a.x<=b.x)&&(a.y!=b.y)&&(a.z<=b.z);

	}
};

struct CV_EXPORTS Sample{
	enum {UNIFORM, NORMAL};
	Sample(	cv::Rect d, int l, int t):dw(d),level(l),genType(t)
	{};

	cv::Rect dw;
    int level;
    int genType;
};

class CV_EXPORTS_W DetectorFast: public Detector{

public:

	// An empty cascade will be created.
    // Param minScale 		is a minimum scale relative to the original size of the image on which cascade will be applied.
    // Param minScale 		is a maximum scale relative to the original size of the image on which cascade will be applied.
    // Param scales 		is a number of scales from minScale to maxScale.
    // Param rejCriteria 	is used for NMS.
    //CV_WRAP DetectorFast(double minScale = 0.4, double maxScale = 5., int scales = 55, int rejCriteria = 1);
	CV_WRAP DetectorFast(ParamDetectorFast parameters);

    CV_WRAP ~DetectorFast();

    // Load soft cascade from FileNode and trace-model.
    // Param fileNode 		is a root node for cascade.
    // Param fileNodeModel 	is a root node for trace-model.
     virtual bool load(const FileNode& cascadeModel,const FileNode& fastModel);
     virtual bool load(const FileNode& cascadeModel);

    // Return the vector of Detection objects (with fast evaluation).
    // Param image is a frame on which detector will be applied.
    // Param rois is a vector of regions of interest. Only the objects that fall into one of the regions will be returned.
    // Param objects is an output array of Detections
    virtual void detectFast(cv::InputArray _image,std::vector<Detection>& objects);
    virtual void detectFastWithMask(cv::InputArray _image, cv::InputArray _mask, double th,std::vector<Detection>& objects);

    void generateSamples(cv::Size imgSize,std::vector<Sample>& samples);

    void setExecParam(uint lastStage,  uint gridSize, double gamma,bool covExp,bool uE, int geomModelType){
    	fastModel.paramDtFast.lastStage=lastStage;
    	fastModel.paramDtFast.gridSize=gridSize;
    	fastModel.paramDtFast.gamma=gamma;
    	fastModel.paramDtFast.covMExpansion=covExp;
    	fastModel.paramDtFast.uniformEnergy=uE;
    	fastModel.paramDtFast.geomModelType=geomModelType;
    };

    static void mergeModels(std::string outPath, std::vector<std::string>& modelsPath){
    	cv::FileStorage outFS(outPath.data(), cv::FileStorage::WRITE);
        if(!outFS.isOpened()){
        	std::cout<<"<<< Error >>> Opening file "<<outPath<<" FAILED !!!";
    		exit(-1);
        }
        std::vector<FastDtModel> models;

        for(uint i=0;modelsPath.size();i++){
        	cv::FileStorage fs(modelsPath[i].data(), cv::FileStorage::READ);
            if(!fs.isOpened()){
        		std::cout<<"<<< Error >>> Opening file "<<modelsPath[i]<<" FAILED !!!";
        		exit(-1);
            }
            FastDtModel model;
           	model.read(fs.getFirstTopLevelNode());
       		models.push_back(model);
       		fs.release();
        }
    	FastDtModel outModel=FastDtModel::mergeModels(models);
      	outFS << cv::softcascade::FastDtModel::ROOT;
       	outModel.write(outFS);
      	outFS.release();
    }

    // Save both models (trace and geometry) in path/Trace_Model.dat and path/Geometry_Model.dat respectively
    void saveModelIntoDat(String path);

    CV_WRAP uint getNumLevels();
private:


    void detectFast_FIXED_GRID(std::vector<Detection>& objects, ChannelStorage& storage, Fields& fld,double pyramidSize);
    void detectFast_GMM(std::vector<Detection>& objects, ChannelStorage& storage, Fields& fld,double pyramidSize);
    void detectFast_GMM_3D(std::vector<Detection>& objects, ChannelStorage& storage, Fields& fld,double pyramidSize);

    // Load trace-model
    CV_WRAP virtual bool loadModel(const FileNode& fastNode);

    // random sampling by normal distribution
    double *r8vec_uniform_01_new ( int n, int *seed);
    double *r8po_fa ( int n, double a[]);
    double *r8vec_normal_01_new ( int n, int *seed);
    double *multinormal_sample( int m, int n, double a[], double mu[], int *seed);

    inline bool isUniformSampling(const cv::Mat& mat){
    	return (mat.at<double>(0,0)==-1. || mat.at<double>(0,1)==-1.);
    };

    // generate samples by Geometric-Model
    void rndSamples(const Level& level,	std::set<Point2i,classPoint2iComp>& dw, cv::Mat& avgM, cv::Mat& covM,const int samplesTot);
    void rndSamples_3D(std::vector<Level>::const_iterator lIt,std::vector<Level>::const_iterator endIt, std::set<Point3i,classPoint3iComp>& dw, cv::Mat& avgM, cv::Mat& covM,const int samplesTot);
    void uniSamples(const FastDtModel::Block& block, const Level& level,
    		int& startX,int& startY, int& endX, int& endY, int& stepX, int& stepY,
    		const int samplesTot);

	struct CV_EXPORTS TempInfo{
    	int rejCriteria;
    	int*	index;
    	int*	weaks;
    };

	TempInfo 	tempI;
	FastDtModel fastModel;

};





// ============================================================================================================== //
//		     Declaration of DetectorTrace (without trace evaluation reduction) and other structures nedded
// ============================================================================================================= //

// Representation of detector results, included subscores and pyramid info
struct CV_EXPORTS Trace{

	// Creates Trace from an object bounding box, confidence and subscores.
	// Param index 			is identifier of each trace in a pyramid
	// Param localMaxIndex  is identifier of local maximum trace that reject me (only if classification==POSITIVE)
	//						Obs: index==localMaxIndex if this trace is local maximum
	// Param localMinIndex  is identifier of lowest trace, only strongs have this information
	//						Obs: if |ROS|=1 than localMinIndex=localMaxIndex
	// Param octaveIndex	is index of octave (inner of pyramid) that contain the positive detection window (the enumeration starts with 0)
	// Param numLevel  		is number of level  (inner of pyramid) that contain the positive detection window
	// Param dw				is detection window
	// Param subScores		is detector result (positive detection window)
	Trace(const uint64 ind,const uint octave, const uint level, const Detection& dw, const std::vector<float>& scores, const std::vector<float>& stagesResp, const int classification);

	enum{NEGATIVE=0,POSITIVE,LOCALMAXIMUM};
	uint64 	index;
	uint64 	localMaxIndex;
	uint64 	localMinIndex;
	uint 	octaveIndex;
	uint 	levelIndex;
	Detection detection;
	std::vector<float> subscores;
	std::vector<float> stages;
	int classType;
};


class CV_EXPORTS_W DetectorTrace: public Detector{

public:

	// An empty cascade will be created.
    // Param minScale 		is a minimum scale relative to the original size of the image on which cascade will be applied.
    // Param minScale 		is a maximum scale relative to the original size of the image on which cascade will be applied.
    // Param scales 		is a number of scales from minScale to maxScale.
    // Param rejCriteria 	is used for NMS.
    //CV_WRAP DetectorTrace(double minScale = 0.4, double maxScale = 5., int scales = 55, int rejCriteria = 1);
    CV_WRAP DetectorTrace(ParamDetectorFast param);

    CV_WRAP ~DetectorTrace();

    // Type of traces to return
    enum{NEGATIVE_TR=0,POSITIVE_TR,LOCALMAXIMUM_TR,NEG_POS_TR};

    // Return the vector of Trace objects.
    // Param image 			is a frame on which detector will be applied.
    // Param rois 			is a vector of regions of interest. Only the objects that fall into one of the regions will be returned.
    // Param positiveTrace 	is an output array of Positive Traces (eventually included local-maxima)
    // Param negativeTrace 	is an output array of Positive Traces
    // Param traceType 		is an output array of Trace
    virtual void detectTrace(InputArray image, InputArray rois, std::vector<Trace>& positiveTrace,std::vector<Trace>& negativeTrace, int traceType=cv::softcascade::DetectorTrace::LOCALMAXIMUM_TR);

    uint nLevels();
    void getInfo4LevelId(int id, cv::Size& workRect, cv::Size& dw,int& octaveId);
    void getRejectionThreshols(std::vector< std::vector<float> >& ths);


private:
    void detectNoRoiTrace(const Mat& image, std::vector<Trace>& positiveTrace,std::vector<Trace>& negativeTrace);
    void detectAtTrace(const int dx, const int dy, const Level& level, const ChannelStorage& storage, std::vector<Trace>& positiveTrace,std::vector<Trace>& negativeTrace,uint levelI);

    // type to traces to return
    int traceType2Return;

};

}}




#endif /* SOFTCASCADE_PLUS_HPP_ */
