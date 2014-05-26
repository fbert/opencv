/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2013, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and / or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "opencv2/softcascade_fast.hpp"
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
#include <iostream> 




const char *const cv::softcascade::FastDtModel::ROOT="Fast_Detector";
const char *const cv::softcascade::FastDtModel::PYRAMID="Pyramid_Setting";
const char *const cv::softcascade::FastDtModel::PYRAMID_MINS="minScale";
const char *const cv::softcascade::FastDtModel::PYRAMID_MAXS="maxScale";
const char *const cv::softcascade::FastDtModel::PYRAMID_NS="nScales";


const char *const cv::softcascade::FastDtModel::TRAININGS="Training_Set";
const char *const cv::softcascade::FastDtModel::TRAININGS_DATAF="datasetFolder";
const char *const cv::softcascade::FastDtModel::TRAININGS_NIMG="numImages";
const char *const cv::softcascade::FastDtModel::TRAININGS_IMGS="imageSize";
const char *const cv::softcascade::FastDtModel::MODELS="Models";

const char *const cv::softcascade::FastDtModel::TraceModel::TRACEMODEL="Trace_Model";
const char *const cv::softcascade::FastDtModel::TraceModel::TRACEMODEL_LASTSTAGES="LastStages";
const char *const cv::softcascade::FastDtModel::TraceModel::TRACEMODEL_LASTSTAGES_LASTS="lastStage";
const char *const cv::softcascade::FastDtModel::TraceModel::TRACEMODEL_LASTSTAGES_SLOPES="slopes";


const char *const cv::softcascade::FastDtModel::GeomModel::GEOMMODEL="Geometry_Model";
const char *const cv::softcascade::FastDtModel::GeomModel::GEOMMODEL_GRIDS="Grids";
const char *const cv::softcascade::FastDtModel::GeomModel::GEOMMODEL_GRID_SIZE="size";
const char *const cv::softcascade::FastDtModel::GeomModel::GEOMMODEL_GRID_BLOCKS="Blocks";
const char *const cv::softcascade::FastDtModel::GeomModel::GEOMMODEL_GRID_BLOCKS_ID="id";
const char *const cv::softcascade::FastDtModel::GeomModel::GEOMMODEL_GRID_BLOCKS_LEVELSH="levelsHist";
const char *const cv::softcascade::FastDtModel::GeomModel::GEOMMODEL_GRID_BLOCKS_LOCATIONSH="locationsHist";
const char *const cv::softcascade::FastDtModel::GeomModel::GEOMMODEL_GRID_BLOCKS_LOCATIONSH_AVG="averages";
const char *const cv::softcascade::FastDtModel::GeomModel::GEOMMODEL_GRID_BLOCKS_LOCATIONSH_COV="covariances";
const char *const cv::softcascade::FastDtModel::GeomModel::GEOMMODEL_GRID_BLOCKS_RECT="rect";
const char *const cv::softcascade::FastDtModel::GeomModel::GEOMMODEL_GRID_BLOCKS_ENERGY="energy";

//------------------ GMM
const char *const cv::softcascade::FastDtModel::GeomModelGMM::GEOMMODELGMM="GMM_Geometry_Model";
const char *const cv::softcascade::FastDtModel::GeomModelGMM::GEOMMODELGMM_LEVELS="Levels";
const char *const cv::softcascade::FastDtModel::GeomModelGMM::GEOMMODELGMM_LEVELS_AVG="averages";
const char *const cv::softcascade::FastDtModel::GeomModelGMM::GEOMMODELGMM_LEVELS_COV="covariances";
const char *const cv::softcascade::FastDtModel::GeomModelGMM::GEOMMODELGMM_LEVELS_MIXC="mixingP";
const char *const cv::softcascade::FastDtModel::GeomModelGMM::GEOMMODELGMM_LEVELS_ENERGY="energy";

//------------------ SEG
const char *const cv::softcascade::FastDtModel::GeomModelSEG::GEOMMODELSEG="SEG_Geometry_Model";
const char *const cv::softcascade::FastDtModel::GeomModelSEG::GEOMMODELSEG_LEVELS="Levels";
const char *const cv::softcascade::FastDtModel::GeomModelSEG::GEOMMODELSEG_LEVELS_UPPERLEFTX="upperLeftX";
const char *const cv::softcascade::FastDtModel::GeomModelSEG::GEOMMODELSEG_LEVELS_UPPERLEFTY="upperLeftY";










// Mnemosyne
cv::softcascade::Detection::Detection(const cv::Rect& b, const float c, int k)
: x(static_cast<ushort>(b.x)), y(static_cast<ushort>(b.y)),
  w(static_cast<ushort>(b.width)), h(static_cast<ushort>(b.height)), confidence(c), kind(k) {}

cv::Rect cv::softcascade::Detection::bb() const
{
    return cv::Rect(x, y, w, h);
}

//namespace {

struct SOctave
{
    SOctave(const int i, const cv::Size& origObjSize, const cv::FileNode& fn)
    : index(i), weaks((int)fn[SC_OCT_WEAKS]), scale((float)std::pow(2,(float)fn[SC_OCT_SCALE])),
      size(cvRound(origObjSize.width * scale), cvRound(origObjSize.height * scale)) {}

    int   index;
    int   weaks;

    float scale;

    cv::Size size;

    static const char *const SC_OCT_SCALE;
    static const char *const SC_OCT_WEAKS;
    static const char *const SC_OCT_SHRINKAGE;
};


struct Weak
{
    Weak(){}
    Weak(const cv::FileNode& fn) : threshold((float)fn[SC_WEAK_THRESHOLD]) {}

    float threshold;

    static const char *const SC_WEAK_THRESHOLD;
};


struct Node
{
    Node(){}
    Node(const int offset, cv::FileNodeIterator& fIt)
    : feature((int)(*(fIt +=2)++) + offset), threshold((float)(*(fIt++))) {}

    int   feature;
    float threshold;
};

struct Feature
{
    Feature() {}
    Feature(const cv::FileNode& fn, bool useBoxes = false) : channel((int)fn[SC_F_CHANNEL])
    {
        cv::FileNode rn = fn[SC_F_RECT];
        cv::FileNodeIterator r_it = rn.begin();

        int x = *r_it++;
        int y = *r_it++;
        int w = *r_it++;
        int h = *r_it++;

        // ToDo: fix me
        if (useBoxes)
            rect = cv::Rect(x, y, w, h);
        else
            rect = cv::Rect(x, y, w + x, h + y);

        // 1 / area
        rarea = 1.f / ((rect.width - rect.x) * (rect.height - rect.y));
    }

    int channel;
    cv::Rect rect;
    float rarea;

    static const char *const SC_F_CHANNEL;
    static const char *const SC_F_RECT;
};

const char *const SOctave::SC_OCT_SCALE      = "scale";
const char *const SOctave::SC_OCT_WEAKS      = "weaks";
const char *const SOctave::SC_OCT_SHRINKAGE  = "shrinkingFactor";
const char *const Weak::SC_WEAK_THRESHOLD   = "treeThreshold";
const char *const Feature::SC_F_CHANNEL     = "channel";
const char *const Feature::SC_F_RECT        = "rect";

struct Level
{
    const SOctave* octave;

    float origScale;
    float relScale;
    int scaleshift;

    cv::Size workRect;
    cv::Size objSize;

    float scaling[2]; // 0-th for channels <= 6, 1-st otherwise

    Level(const SOctave& oct, const float scale, const int shrinkage, const int w, const int h)
    :  octave(&oct), origScale(scale), relScale(scale / oct.scale),
       workRect(cv::Size(cvRound(w / (float)shrinkage),cvRound(h / (float)shrinkage))),
       objSize(cv::Size(cvRound(oct.size.width * relScale), cvRound(oct.size.height * relScale)))
    {
        scaling[0] = ((relScale >= 1.f)? 1.f : (0.89f * std::pow(relScale, 1.099f / std::log(2.f)))) / (relScale * relScale);
        scaling[1] = 1.f;
        scaleshift = static_cast<int>(relScale * (1 << 16));
    }

    void addDetection(const int x, const int y, float confidence, std::vector<cv::softcascade::Detection>& detections) const
    {
        // fix me
        int shrinkage = 4;//(*octave).shrinkage;
        cv::Rect rect(cvRound(x * shrinkage), cvRound(y * shrinkage), objSize.width, objSize.height);

        detections.push_back(cv::softcascade::Detection(rect, confidence));
    }

    float rescale(cv::Rect& scaledRect, const float threshold, int idx) const
    {
#define SSHIFT(a) ((a) + (1 << 15)) >> 16
        // rescale
        scaledRect.x      = SSHIFT(scaleshift * scaledRect.x);
        scaledRect.y      = SSHIFT(scaleshift * scaledRect.y);
        scaledRect.width  = SSHIFT(scaleshift * scaledRect.width);
        scaledRect.height = SSHIFT(scaleshift * scaledRect.height);
#undef SSHIFT
        float sarea = static_cast<float>((scaledRect.width - scaledRect.x) * (scaledRect.height - scaledRect.y));

        // compensation areas rounding
        return (sarea == 0.0f)? threshold : (threshold * scaling[idx] * sarea);
    }
};
struct ChannelStorage
{
    cv::Mat hog;
    int shrinkage;
    int offset;
    size_t step;
    int model_height;

    cv::Ptr<cv::softcascade::ChannelFeatureBuilder> builder;

    enum {HOG_BINS = 6, HOG_LUV_BINS = 10};

    ChannelStorage(const cv::Mat& colored, int shr, cv::String featureTypeStr) : shrinkage(shr)
    {
        model_height = cvRound(colored.rows / (float)shrinkage);
        if (featureTypeStr == "ICF") featureTypeStr = "HOG6MagLuv";

        builder = cv::softcascade::ChannelFeatureBuilder::create(featureTypeStr);
        (*builder)(colored, hog, cv::Size(cvRound(colored.cols / (float)shrinkage), model_height));

        step = hog.step1();
    }

    float get(const int channel, const cv::Rect& area) const
    {
        const int *ptr = hog.ptr<const int>(0) + model_height * channel * step + offset;

        int a = ptr[area.y      * step + area.x];
        int b = ptr[area.y      * step + area.width];
        int c = ptr[area.height * step + area.width];
        int d = ptr[area.height * step + area.x];

        return static_cast<float>(a - b + c - d);
    }
};

//}

struct cv::softcascade::Detector::Fields
{
    float minScale;
    float maxScale;
    int scales;

    int origObjWidth;
    int origObjHeight;

    int shrinkage;

    std::vector<SOctave> octaves;
    std::vector<Weak>    weaks;
    std::vector<Node>    nodes;
    std::vector<float>   leaves;
    std::vector<Feature> features;

    std::vector<Level> levels;

    cv::Size frameSize;

    typedef std::vector<SOctave>::iterator  octIt_t;
    typedef std::vector<Detection> dvector;

    String featureTypeStr;

    void detectAt(const int dx, const int dy, const Level& level, const ChannelStorage& storage, dvector& detections) const
    {
        float detectionScore = 0.f;

        const SOctave& octave = *(level.octave);

        int stBegin = octave.index * octave.weaks, stEnd = stBegin + octave.weaks;


        for(int st = stBegin; st < stEnd; ++st)
        {
            const Weak& weak = weaks[st];

            int nId = st * 3;

            // work with root node
            const Node& node = nodes[nId];
            const Feature& feature = features[node.feature];

            cv::Rect scaledRect(feature.rect);

            float threshold = level.rescale(scaledRect, node.threshold, (int)(feature.channel > 6)) * feature.rarea;
            float sum = storage.get(feature.channel, scaledRect);
            int next = (sum >= threshold)? 2 : 1;

            // leaves
            const Node& leaf = nodes[nId + next];
            const Feature& fLeaf = features[leaf.feature];

            scaledRect = fLeaf.rect;
            threshold = level.rescale(scaledRect, leaf.threshold, (int)(fLeaf.channel > 6)) * fLeaf.rarea;
            sum = storage.get(fLeaf.channel, scaledRect);

            int lShift = (next - 1) * 2 + ((sum >= threshold) ? 1 : 0);
            float impact = leaves[(st * 4) + lShift];

            detectionScore += impact;

            if (detectionScore <= weak.threshold) return;
        }

        if (detectionScore > 0)
            level.addDetection(dx, dy, detectionScore, detections);
    }

    octIt_t fitOctave(const float& logFactor)
    {
        float minAbsLog = FLT_MAX;
        octIt_t res =  octaves.begin();
        for (octIt_t oct = octaves.begin(); oct < octaves.end(); ++oct)
        {
            const SOctave& octave =*oct;
            float logOctave = std::log(octave.scale);
            float logAbsScale = fabs(logFactor - logOctave);

            if(logAbsScale < minAbsLog)
            {
                res = oct;
                minAbsLog = logAbsScale;
            }
        }
        return res;
    }

    // compute levels of full pyramid
    void calcLevels(const cv::Size& curr, float mins, float maxs, int total)
    {
        if (frameSize == curr && maxs == maxScale && mins == minScale && total == scales) return;

        frameSize = curr;
        maxScale = maxs; minScale = mins; scales = total;
        CV_Assert(scales > 1);

        levels.clear();
        float logFactor = (std::log(maxScale) - std::log(minScale)) / (scales -1);

        float scale = minScale;
        for (int sc = 0; sc < scales; ++sc)
        {
            int width  = static_cast<int>(std::max(0.0f, frameSize.width  - (origObjWidth  * scale)));
            int height = static_cast<int>(std::max(0.0f, frameSize.height - (origObjHeight * scale)));

            float logScale = std::log(scale);
            octIt_t fit = fitOctave(logScale);


            Level level(*fit, scale, shrinkage, width, height);

            if (!width || !height)
                break;
            else
                levels.push_back(level);

            if (fabs(scale - maxScale) < FLT_EPSILON) break;
            scale = std::min(maxScale, expf(std::log(scale) + logFactor));
        }
    }

    bool fill(const FileNode &root)
    {
        // cascade properties
        static const char *const SC_STAGE_TYPE       = "stageType";
        static const char *const SC_BOOST            = "BOOST";

        static const char *const SC_FEATURE_TYPE     = "featureType";
        static const char *const SC_HOG6_MAG_LUV     = "HOG6MagLuv";
        static const char *const SC_ICF              = "ICF";

        static const char *const SC_ORIG_W           = "width";
        static const char *const SC_ORIG_H           = "height";

        static const char *const SC_OCTAVES          = "octaves";
        static const char *const SC_TREES            = "trees";
        static const char *const SC_FEATURES         = "features";

        static const char *const SC_INTERNAL         = "internalNodes";
        static const char *const SC_LEAF             = "leafValues";

        static const char *const SC_SHRINKAGE        = "shrinkage";

        static const char *const FEATURE_FORMAT      = "featureFormat";

        // only Ada Boost supported
        String stageTypeStr = (String)root[SC_STAGE_TYPE];
        CV_Assert(stageTypeStr == SC_BOOST);

        String fformat = (String)root[FEATURE_FORMAT];
        bool useBoxes = (fformat == "BOX");

        // only HOG-like integral channel features supported
        featureTypeStr = (String)root[SC_FEATURE_TYPE];
        CV_Assert(featureTypeStr == SC_ICF || featureTypeStr == SC_HOG6_MAG_LUV);

        origObjWidth  = (int)root[SC_ORIG_W];
        origObjHeight = (int)root[SC_ORIG_H];

        shrinkage = (int)root[SC_SHRINKAGE];

        FileNode fn = root[SC_OCTAVES];
        if (fn.empty()) return false;

        // for each octave
        FileNodeIterator it = fn.begin(), it_end = fn.end();
        for (int octIndex = 0; it != it_end; ++it, ++octIndex)
        {
            FileNode fns = *it;
            SOctave octave(octIndex, cv::Size(origObjWidth, origObjHeight), fns);
            CV_Assert(octave.weaks > 0);
            octaves.push_back(octave);

            FileNode ffs = fns[SC_FEATURES];
            if (ffs.empty()) return false;

            fns = fns[SC_TREES];
            if (fn.empty()) return false;

            FileNodeIterator st = fns.begin(), st_end = fns.end();
            for (; st != st_end; ++st )
            {
                weaks.push_back(Weak(*st));

                fns = (*st)[SC_INTERNAL];
                FileNodeIterator inIt = fns.begin(), inIt_end = fns.end();
                for (; inIt != inIt_end;)
                    nodes.push_back(Node((int)features.size(), inIt));

                fns = (*st)[SC_LEAF];
                inIt = fns.begin(), inIt_end = fns.end();

                for (; inIt != inIt_end; ++inIt)
                    leaves.push_back((float)(*inIt));
            }

            st = ffs.begin(), st_end = ffs.end();
            for (; st != st_end; ++st )
                features.push_back(Feature(*st, useBoxes));
        }

        return true;
    }
};

cv::softcascade::Detector::Detector(const double mins, const double maxs, const int nsc, const int rej)
: fields(0), minScale(mins), maxScale(maxs), scales(nsc), rejCriteria(rej) {}

cv::softcascade::Detector::~Detector() { delete fields;}

void cv::softcascade::Detector::read(const cv::FileNode& fn)
{
    Algorithm::read(fn);
}

bool cv::softcascade::Detector::load(const cv::FileNode& fn)
{
    if (fields) delete fields;

    fields = new Fields;
    return fields->fill(fn);
}

namespace {

using cv::softcascade::Detection;
typedef std::vector<Detection>  dvector;


struct ConfidenceGt
{
    bool operator()(const Detection& a, const Detection& b) const
    {
        return a.confidence > b.confidence;
    }
};

static float overlap(const cv::Rect &a, const cv::Rect &b)
{
    int w = std::min(a.x + a.width,  b.x + b.width)  - std::max(a.x, b.x);
    int h = std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y);

    return (w < 0 || h < 0)? 0.f : (float)(w * h);
}

void DollarNMS(dvector& objects)
{
    static const float DollarThreshold = 0.65f;
    std::sort(objects.begin(), objects.end(), ConfidenceGt());

    for (dvector::iterator dIt = objects.begin(); dIt != objects.end(); ++dIt)
    {
        const Detection &a = *dIt;
        for (dvector::iterator next = dIt + 1; next != objects.end(); )
        {
            const Detection &b = *next;

            const float ovl =  overlap(a.bb(), b.bb()) / std::min(a.bb().area(), b.bb().area());

            if (ovl > DollarThreshold)
                next = objects.erase(next);
            else
                ++next;
        }
    }
}

static void suppress(int type, std::vector<Detection>& objects)
{
    CV_Assert(type == cv::softcascade::Detector::DOLLAR);
    DollarNMS(objects);
}

}

void cv::softcascade::Detector::detectNoRoi(const cv::Mat& image, std::vector<Detection>& objects) const
{
    Fields& fld = *fields;
    // create integrals
    ChannelStorage storage(image, fld.shrinkage, fld.featureTypeStr);

    typedef std::vector<Level>::const_iterator lIt;
    for (lIt it = fld.levels.begin(); it != fld.levels.end(); ++it)
    {
        const Level& level = *it;

        // we train only 3 scales.
        if (level.origScale > 2.5) break;

        for (int dy = 0; dy < level.workRect.height; ++dy)
        {
            for (int dx = 0; dx < level.workRect.width; ++dx)
            {
                storage.offset = (int)(dy * storage.step + dx);
                fld.detectAt(dx, dy, level, storage, objects);
            }
        }
    }

    if (rejCriteria != NO_REJECT) suppress(rejCriteria, objects);
}

void cv::softcascade::Detector::detect(cv::InputArray _image, cv::InputArray _rois, std::vector<Detection>& objects) const
{
    // only color images are suppered
    cv::Mat image = _image.getMat();
    CV_Assert(image.type() == CV_8UC3);

    Fields& fld = *fields;
    fld.calcLevels(image.size(),(float) minScale, (float)maxScale, scales);

    objects.clear();

    if (_rois.empty())
        return detectNoRoi(image, objects);

    int shr = fld.shrinkage;

    cv::Mat roi = _rois.getMat();
    cv::Mat mask(image.rows / shr, image.cols / shr, CV_8UC1);

    mask.setTo(cv::Scalar::all(0));
    cv::Rect* r = roi.ptr<cv::Rect>(0);
    for (int i = 0; i < (int)roi.cols; ++i)
        cv::Mat(mask, cv::Rect(r[i].x / shr, r[i].y / shr, r[i].width / shr , r[i].height / shr)).setTo(cv::Scalar::all(1));

    // create integrals
    ChannelStorage storage(image, shr, fld.featureTypeStr);

    typedef std::vector<Level>::const_iterator lIt;
    for (lIt it = fld.levels.begin(); it != fld.levels.end(); ++it)
    {
         const Level& level = *it;

        // we train only 3 scales.
        if (level.origScale > 2.5) break;

         for (int dy = 0; dy < level.workRect.height; ++dy)
         {
             uchar* m  = mask.ptr<uchar>(dy);
             for (int dx = 0; dx < level.workRect.width; ++dx)
             {
                 if (m[dx])
                 {
                     storage.offset = (int)(dy * storage.step + dx);
                     fld.detectAt(dx, dy, level, storage, objects);
                 }
             }
         }
    }

    if (rejCriteria != NO_REJECT) suppress(rejCriteria, objects);
}

void cv::softcascade::Detector::detect(InputArray _image, InputArray _rois,  OutputArray _rects, OutputArray _confs) const
{
    std::vector<Detection> objects;
    detect( _image, _rois, objects);

    _rects.create(1, (int)objects.size(), CV_32SC4);
    cv::Mat_<cv::Rect> rects = (cv::Mat_<cv::Rect>)_rects.getMat();
    cv::Rect* rectPtr = rects.ptr<cv::Rect>(0);

    _confs.create(1, (int)objects.size(), CV_32F);
    cv::Mat confs = _confs.getMat();
    float* confPtr = confs.ptr<float>(0);

    typedef std::vector<Detection>::const_iterator IDet;

    int i = 0;
    for (IDet it = objects.begin(); it != objects.end(); ++it, ++i)
    {
        rectPtr[i] = (*it).bb();
        confPtr[i] = (*it).confidence;
    }
}

//************************NEW CLASSES IMPLEMENTED BY Federico Bartoli*****************************************************


// ============================================================================================================== //
//		     Implementation of DetectorFast (with trace evaluation reduction)
// ============================================================================================================= //




void cv::softcascade::FastDtModel::TraceModel::compute(uint levels){

	slopes.clear();

	double slopeSum;

	for(LinesMap::iterator itS=linesParam.begin();itS!=linesParam.end();++itS){
		slopes[itS->first]=std::vector<double>(levels,0.);
		for(LevelsMap::iterator itL=itS->second.begin();itL!=itS->second.end();++itL){
			slopeSum=0.;
			for (std::vector<Vec4f>::iterator it=itL->second.begin();it!=itL->second.end();++it)
				slopeSum+=it->val[Vy]/it->val[Vx];
			slopes[itS->first][itL->first]=slopeSum/(double)itL->second.size();
		}
	}
}
void cv::softcascade::FastDtModel::TraceModel::write(FileStorage& fso) const{

	fso << TraceModel::TRACEMODEL_LASTSTAGES<<  "[";


	SlopesMap::const_iterator itS= slopes.begin();
	for( ;itS!=slopes.end();++itS){
		fso << "{";
		fso<< TraceModel::TRACEMODEL_LASTSTAGES_LASTS << (int)itS->first;
		fso<< TraceModel::TRACEMODEL_LASTSTAGES_SLOPES << "[";
		for(std::vector<double >::const_iterator itL=itS->second.begin();itL!=itS->second.end();++itL)
			fso<< *itL;
		fso << "]";
		fso << "}";
	}
	fso<< "]";

}
void cv::softcascade::FastDtModel::TraceModel::read(const FileNode& node){


	FileNode lastStages = node[TRACEMODEL_LASTSTAGES];

	slopes.clear();

	for(FileNodeIterator itS=lastStages.begin();itS!=lastStages.end();++itS){
		int stage=(int)(*itS)[TRACEMODEL_LASTSTAGES_LASTS];
		FileNode slopesN = (*itS)[TRACEMODEL_LASTSTAGES_SLOPES];

		slopesN >> slopes[stage];

		/*for(FileNodeIterator itSl=slopesN.begin();itSl!=slopesN.end();++itSl)
			slopes[stage][itSl-slopesN.begin()]=(double)(*itSl);*/
	}
	linesParam.clear();
}

void cv::softcascade::FastDtModel::GeomModel::write(FileStorage& fso) const{

	fso << GEOMMODEL_GRIDS <<  "[";


	Grids::const_iterator itG=grids.begin();
	for( ;itG!=grids.end();++itG){
		fso << "{";
		fso<< GEOMMODEL_GRID_SIZE << (int)itG->first;
		fso << GEOMMODEL_GRID_BLOCKS << "[";
		uint id=0;
		for(std::vector<Block>::const_iterator itB=itG->second.begin();itB!=itG->second.end();++itB){
			fso<<"{";
			fso<< GEOMMODEL_GRID_BLOCKS_ID <<(int) id++;
			fso<< GEOMMODEL_GRID_BLOCKS_LEVELSH	<< itB->levelsHist;
			fso<< GEOMMODEL_GRID_BLOCKS_LOCATIONSH;
			fso<<"{";

			fso<< GEOMMODEL_GRID_BLOCKS_LOCATIONSH_AVG << "[";
			for(std::vector<AverageCov>::const_iterator itA=itB->locationsHist.begin();itA!=itB->locationsHist.end();++itA)
				fso << itA->avg;
			fso<<"]";

			fso<< GEOMMODEL_GRID_BLOCKS_LOCATIONSH_COV << "[";
			for(std::vector<AverageCov>::const_iterator itC=itB->locationsHist.begin();itC!=itB->locationsHist.end();++itC)
				fso << itC->cov;
			fso<<"]";

			fso<<"}";
			fso<<  GEOMMODEL_GRID_BLOCKS_RECT << itB->rect;
			fso<<  GEOMMODEL_GRID_BLOCKS_ENERGY	<< itB->energy;
			fso<<"}";
		}
		fso<< "]";
		fso << "}";
	}

	fso<< "]";

}
void cv::softcascade::FastDtModel::GeomModel::read(const FileNode& node){


	FileNode gridsNode = node[GEOMMODEL_GRIDS];

	grids.clear();

	for(FileNodeIterator itG=gridsNode.begin();itG!=gridsNode.end();++itG){

		int gridSize=(int)(*itG)[GEOMMODEL_GRID_SIZE];
		FileNode blocksNode = (*itG)[GEOMMODEL_GRID_BLOCKS];

		grids[gridSize]=std::vector<Block>();
		for(FileNodeIterator itB=blocksNode.begin();itB!=blocksNode.end();++itB){

			std::vector<double> levelsHist;
			(*itB)[GEOMMODEL_GRID_BLOCKS_LEVELSH] >> levelsHist;

    		std::vector<AverageCov> locationsHist;
    		FileNode avgNode =(*itB)[GEOMMODEL_GRID_BLOCKS_LOCATIONSH][GEOMMODEL_GRID_BLOCKS_LOCATIONSH_AVG];
    		FileNode covNode=(*itB)[GEOMMODEL_GRID_BLOCKS_LOCATIONSH][GEOMMODEL_GRID_BLOCKS_LOCATIONSH_COV];
    		for(FileNodeIterator itAvg=avgNode.begin(), itCov=covNode.begin(); itAvg!=avgNode.end();++itAvg,++itCov){
    			Mat avgM(1,2,CV_64FC1);
    			Mat covM(2,2,CV_64FC1);

    			(*itAvg) >> avgM;
    			(*itCov) >> covM;
    			//std::cout<<"Cov: "<<covM.at<double>(0,0)<<","<<covM.at<double>(0,1)<<
    			//		covM.at<double>(1,0)<<","<<covM.at<double>(1,1)<<")"<<std::endl;

    			locationsHist.push_back(AverageCov(avgM,covM));
    		}

    		Rect rect;
    		(*itB)[GEOMMODEL_GRID_BLOCKS_RECT] >> rect;

    		double	energy=(double)(*itB)[GEOMMODEL_GRID_BLOCKS_ENERGY];

			grids[gridSize].push_back(Block(levelsHist,locationsHist,rect,energy));

		}
	}
}
void cv::softcascade::FastDtModel::GeomModel::setUniform(Size imgSize,uint levels){

	// initialize blocks, levels and energy histograms for each grid size

	for(uint g=0;g<gridsSize.size();g++){
		grids[gridsSize[g]]=std::vector<Block>(gridsSize[g]*gridsSize[g],Block(levels, 1./(double)(levels),1./(double)(gridsSize[g]*gridsSize[g])));

		// compute rect for each block
		uint dw=  imgSize.width/gridsSize[g];
		uint dh= imgSize.height/gridsSize[g];

		std::cout<<"Grid Size: " <<gridsSize[g]<<"x"<<gridsSize[g] <<std::endl;

		uint count=0;
		for(uint b_y=0,y=0;b_y<gridsSize[g];b_y++,y+=dh+1){
			for(uint b_x=0,x=0;b_x<gridsSize[g];b_x++, x+=dw+1){
				std::cout<<"\t Block " <<count<<":";
				std::cout<<" Size - rect("<<x<<","<<y<<","<<dw<<","<<dh<<")"<<std::endl;
				grids[gridsSize[g]][count++].rect=Rect(x,y,dw,dh);


				std::cout<<"\t\t Levels Histogram: ";
				for(uint i=0;i<grids[gridsSize[g]][count-1].levelsHist.size();i++)
					std::cout<<grids[gridsSize[g]][count-1].levelsHist[i]<<" ";
				std::cout<<std::endl;
			}
		}
	}
}

void cv::softcascade::FastDtModel::GeomModel::compute(Size imgSize,uint levels){

	typedef std::map<uint,std::vector<std::vector<std::vector<double> > > > Locations;
	Locations strongLoc;

	std::map<uint,double> energyTot;

	std::cout<<"---- Geometric Model:  Frame Splits ----" <<std::endl;

	// initialize block and levels and energy histograms
	for(uint g=0;g<gridsSize.size();g++){
		grids[gridsSize[g]]=std::vector<Block>(gridsSize[g]*gridsSize[g],Block(levels));
		strongLoc[gridsSize[g]]= std::vector<std::vector<std::vector<double> > >  (gridsSize[g]*gridsSize[g],
				std::vector<std::vector<double> >(levels,std::vector<double>()));

		// compute rect for each block
		uint dw= cvRound((double)(imgSize.width)/gridsSize[g]);
		uint dh= cvRound((double)(imgSize.height)/gridsSize[g]);


		std::cout<<"Grid Size: " <<gridsSize[g]<<"x"<<gridsSize[g] <<std::endl;

		uint count=0;
		for(uint b_y=0,y=0;b_y<gridsSize[g];b_y++,y+=dh+1){
			for(uint b_x=0,x=0;b_x<gridsSize[g];b_x++, x+=dw+1){
				std::cout<<"\t Size Block "<<count<<": rect("<<x<<","<<y<<","<<dw<<","<<dh<<")"<<std::endl;
				grids[gridsSize[g]][count++].rect=Rect(x,y,dw,dh);

			}
		}
	}


	std::cout<<"\n---- Geometric Model:  Assign strong locations for each block-levels ----" <<std::endl;
	// extraction statistics of strongs for all grids size: levels histogram (not-normalized) and locations
	for(StrongsROI::iterator l_s=upperLeftPonts.begin();l_s!=upperLeftPonts.end();++l_s){
		uint level=l_s->first;

		for(std::vector<StrongROI>::iterator s=l_s->second.begin();s!=l_s->second.end();++s){
			for(Grids::iterator g=grids.begin();g!=grids.end();++g){
				for(std::vector<Block>::iterator b=g->second.begin();b!=g->second.end();++b){
					if(b->rect.contains(Point(s->dw.x+cvRound((double)(s->dw.width)/2),
											  s->dw.y+cvRound((double)(s->dw.height)/2)))){
						b->levelsHist[level]+=s->rank;
						std::cout<<"\t Add strong ("<<s->dw.x<<","<<s->dw.y<<","<<s->dw.width<<","<<s->dw.height<<") in: "
								<<" G- "<< g->first
								<<" B- "<< b-g->second.begin()
								<<" L- "<<level
								<<std::endl;

						// insert location (with unique check) in the order: x y
						bool duplicate=false;
						for(uint xI=0;xI<strongLoc[g->first][b-g->second.begin()][level].size();xI=xI+2){
							if(strongLoc[g->first][b-g->second.begin()][level][xI]== (double)(s->dw.x) &&
							   strongLoc[g->first][b-g->second.begin()][level][xI+1]== (double)(s->dw.y)){

								duplicate=true;
								break;
							}
						}
						if(!duplicate){
							strongLoc[g->first][b-g->second.begin()][level].push_back((double)(s->dw.x));
							strongLoc[g->first][b-g->second.begin()][level].push_back((double)(s->dw.y));
						}

						break;
					}
				}
			}
		}
	}

	std::cout<<"---- Geometric Model:  Computation Average and Covariance Matrix ----" <<std::endl;
	// computation average and covariance position for each block_level
	for(Grids::iterator g=grids.begin();g!=grids.end();++g){
		energyTot[g->first]=0.;
		std::cout<<"Grid " <<g->first<<"x"<<g->first <<std::endl;
		for(std::vector<Block>::iterator b=g->second.begin();b!=g->second.end();++b){
			std::cout<<"\t Block " <<b-g->second.begin() <<std::endl;
			energyTot[g->first]+=std::accumulate(b->levelsHist.begin(),b->levelsHist.end(),0.);


			// compute average and covariance
			for(uint level=0;level<levels;level++){
				std::cout<<"Level " <<level<<": "<<std::endl;
				uint nStrong=(uint)((double) (strongLoc[g->first][b-g->second.begin()][level].size())/2);


				// NO STRONGS
				if(nStrong<=2){
					std::cout<<"\t<<Strongs detected for this level <3 >>\t USE UNIFORM EXTRACTION"<<std::endl;
					b->locationsHist[level].avg=Mat(1,2,CV_64FC1,-1.);
					b->locationsHist[level].cov=Mat(2,2,CV_64FC1,0.);
				}
				else{
					std::cout<<"\t<<Multiple strongs detected for this level>>"<<std::endl;
					Mat positions(nStrong,2,CV_64FC1, strongLoc[g->first][b-g->second.begin()][level].data());


					for(int row=0;row<positions.rows;row++){
						std::cout<< positions.at<Vec2d>(row,0)[0]<<","<<positions.at<Vec2d>(row,0)[1]<<std::endl;
					}
					b->locationsHist[level].cov=Mat(2,2,CV_64FC1);
					b->locationsHist[level].avg=Mat(1,2,CV_64FC1);
					try{
						// CV_COVAR_NORMAL=1, CV_COVAR_ROWS=8
						calcCovarMatrix(positions, b->locationsHist[level].cov,b->locationsHist[level].avg, COVAR_NORMAL | COVAR_ROWS | COVAR_SCALE,CV_64FC1);
						// Cov. Matrix NOT POSITIVE DEF
						std::cout<<"\t Det:"<<FastDtModel::det(b->locationsHist[level].cov)<<std::endl;


						if(FastDtModel::det(b->locationsHist[level].cov)<=0.){
							std::cout<<"\t<<<<Cov not definite positive>>>>  PERTURBATION ALL OBSERVATION"<<std::endl;

								RNG rng;

								Mat rngM(positions.size(),positions.type(),0.);
								rng.fill(rngM, RNG::UNIFORM, 0., 5.);

								positions+=rngM;
								calcCovarMatrix(positions, b->locationsHist[level].cov,b->locationsHist[level].avg, COVAR_NORMAL | COVAR_ROWS,CV_64FC1);
								if(FastDtModel::det(b->locationsHist[level].cov)<=0.){
									std::cout<<"<<<Also with perturbation the Cov. Matrix remains not definite positive>>>  EXCEPTION";
									throw;
							}

						}
					}
					catch (Exception& e) {
						b->locationsHist[level].avg=Mat(1,2,CV_64FC1,-1.);
						b->locationsHist[level].cov=Mat(2,2,CV_64FC1,0.);
						std::cout<<"<<<< Exception in covariance computation!!! >>>\t DEFAULT INIT."<<std::endl;
					}
				}
				std::cout<<"\t\t Average Matrix: "<<b->locationsHist[level].avg.at<double>(0,0)<< " "
						<<b->locationsHist[level].avg.at<double>(0,1)<< std::endl;

				std::cout<<"\t\t Covariance Matrix: "<< b->locationsHist[level].cov.at<double>(0,0)<< " "
					     <<b->locationsHist[level].cov.at<double>(0,1)<<";"
					     <<b->locationsHist[level].cov.at<double>(1,0)<<" "
					     <<b->locationsHist[level].cov.at<double>(1,1)<<std::endl;
			}
		}
	}


	std::cout<<"---- Geometric Model:  Energy computation and levels histogram normalization ----" <<std::endl;
	// computation energy for each block and normalization of levels histogram
	for(Grids::iterator g=grids.begin();g!=grids.end();++g){
		std::cout<<"Grid " <<g->first<<"x"<<g->first <<std::endl;
		for(std::vector<Block>::iterator b=g->second.begin();b!=g->second.end();++b){
			std::cout<<"\t Block " <<b-g->second.begin() <<std::endl;
			double levHistAcc=std::accumulate(b->levelsHist.begin(),b->levelsHist.end(),0.);

			// energy
			b->energy=levHistAcc/energyTot[g->first];

			// levels histogram normalization
			if(levHistAcc>0){
				for(uint l=0;l<levels;l++)
					b->levelsHist[l]/=levHistAcc;
			}
			std::cout<<"\t Levels Histogram: ";
			for(uint i=0;i<b->levelsHist.size();i++)
				std::cout<<b->levelsHist[i]<<" ";
			std::cout<<std::endl;
		}
	}
}

void cv::softcascade::FastDtModel::GeomModelGMM::read(const FileNode& node){


	FileNode levelsNode = node[GEOMMODELGMM_LEVELS];

	gmm.clear();

	std::cout<<"Reading GMM model... "<<std::endl;

	for(FileNodeIterator itL=levelsNode.begin();itL!=levelsNode.end();++itL){

		uint levelId=itL-levelsNode.begin();

		gmm.push_back(FastDtModel::MixtureC());

		if( (*itL).empty()){
			continue;
		}


		FileNode avgsNode = (*itL)[GEOMMODELGMM_LEVELS_AVG];
		FileNode covsNode = (*itL)[GEOMMODELGMM_LEVELS_COV];

		(*itL)[GEOMMODELGMM_LEVELS_MIXC] >> gmm[levelId].mixingP;
		(*itL)[GEOMMODELGMM_LEVELS_ENERGY] >> gmm[levelId].energy;

		for(FileNodeIterator itA=avgsNode.begin(),itC=covsNode.begin() ;itA!=avgsNode.end();++itA,++itC){

			Mat avgM(1,2,CV_64FC1);
			Mat covM(2,2,CV_64FC1);

			(*itA) >> avgM;
			(*itC) >> covM;

			gmm[levelId].avgCov.push_back(AverageCov(avgM,covM));
		}
	}
}
void cv::softcascade::FastDtModel::GeomModelSEG::read(const FileNode& node){


	FileNode levelsNode = node[GEOMMODELSEG_LEVELS];

	upperLeftPoints.clear();

	std::cout<<"Reading SEG model... ";

	for(FileNodeIterator itL=levelsNode.begin();itL!=levelsNode.end();++itL){

		uint levelId=itL-levelsNode.begin();


		std::set<Point2i,classPoint2iComp> dw;
		upperLeftPoints.push_back(dw);


		if( (*itL).empty()){
			continue;
		}


		Mat upperLeftXM;
		Mat upperLeftYM;
		((*itL)[GEOMMODELSEG_LEVELS_UPPERLEFTX]) >> upperLeftXM;
		((*itL)[GEOMMODELSEG_LEVELS_UPPERLEFTY]) >> upperLeftYM;


		std::cout<<"Level:"<<levelId<<"-"<<upperLeftXM.cols<<std::endl;


		if (upperLeftXM.cols!=upperLeftYM.cols){
			std::cerr<<"error: Different size of X and Y points ";

		}

		for(int col=0; col<upperLeftXM.cols;col++){


			double x = upperLeftXM.at<double>(0,col);
			double y= upperLeftYM.at<double>(0,col);

			if (x<0 || y<0){
				continue;
			}
			upperLeftPoints[levelId].insert(Point2i(cvRound((x)/4), cvRound((y)/4)));
		}



	}
}






bool cv::softcascade::FastDtModel::getSlopeAt(uint stage,uint level,double& slope){
	try{
		slope= traceModel.slopes.at(stage)[level];
		return true;
	}
	catch (Exception& e) {
		return false;
	}
}

void cv::softcascade::FastDtModel::getLastSt(std::vector<uint>& stages){
	stages.clear();

	for(TraceModel::SlopesMap::iterator itS=traceModel.slopes.begin();itS!=traceModel.slopes.end();++itS)
		stages.push_back(itS->first);

}

bool cv::softcascade::FastDtModel::getLevelsForStage(uint  lastStage, std::vector<uint>& levels){
	levels.clear();

	try{
		for(uint l=0;l<traceModel.slopes.size();l++)
			levels.push_back(l);
		return true;
	}
	catch (Exception& e) {
		return false;
	}
}

cv::softcascade::ParamDetectorFast::ParamDetectorFast()
: minScale(0.4) , maxScale(5.), nScales(55), nMS(1),lastStage(512), gridSize(2), gamma(0.063), round(3), covMExpansion(true)
{

}
cv::softcascade::ParamDetectorFast::ParamDetectorFast(double minS, double maxS, uint nS, int noMS, uint lastSt, uint gridS,double gam, uint rd,bool covExp,bool uE)
: minScale(minS) , maxScale(maxS), nScales(nS), nMS(noMS),lastStage(lastSt), gridSize(gridS), gamma(gam), round(rd), covMExpansion(covExp), uniformEnergy(uE), geomModelType(0)
{
}


cv::softcascade::FastDtModel::FastDtModel()
{	octaves.clear();
	levels.clear();
}
cv::softcascade::FastDtModel::FastDtModel(ParamDetectorFast param, std::string data="",uint numImg=-1, Size imgS=Size(0,0))
: paramDtFast(param), dataset(data), numImages(numImg),imgSize(imgS)
{	octaves.clear();
	levels.clear();
}


void cv::softcascade::FastDtModel::FastDtModel::write(cv::FileStorage& fso) const{

	fso<<"{";
		// Pyramid Section
		fso<< PYRAMID<< "{";
			fso<< PYRAMID_MINS << paramDtFast.minScale;
			fso<< PYRAMID_MAXS << paramDtFast.maxScale;
			fso<< PYRAMID_NS  << (int) paramDtFast.nScales;
		fso<< "}";

		// Training Set Section
		fso<< TRAININGS << "{";
			fso<< TRAININGS_DATAF<< dataset.data();
			fso<< TRAININGS_NIMG << (int) numImages;
			fso<< TRAININGS_IMGS << imgSize;
		fso<< "}";

		// Models
		fso<< MODELS<< "{";
			// Trace-model Section
			fso<<TraceModel::TRACEMODEL<< "{";
			traceModel.write(fso);
			fso<< "}";

			// Geometry-model Sction
			fso<<GeomModel::GEOMMODEL<< "{";
			geomModel.write(fso);
			fso<< "}";
		fso<< "}";

	fso<< "}";
}
void cv::softcascade::FastDtModel::FastDtModel::read(const cv::FileNode& node){

	//traceModel.read(node[MODELS][TraceModel::TRACEMODEL]);

	// GRID Model
	if (!(node[MODELS][GeomModel::GEOMMODEL].empty()))
		geomModel.read(node[MODELS][GeomModel::GEOMMODEL]);
	// GMM Model
	if (!(node[MODELS][GeomModelGMM::GEOMMODELGMM].empty()))
		geomModelGMM.read(node[MODELS][GeomModelGMM::GEOMMODELGMM]);

	// SEG MODEL
	if (!(node[MODELS][GeomModelSEG::GEOMMODELSEG].empty()))
		geomModelSEG.read(node[MODELS][GeomModelSEG::GEOMMODELSEG]);

}


void cv::softcascade::FastDtModel::addTraceForTraceModel(uint stage,uint level,const std::vector<Point2d>& trace){


	Vec4f line; //(vx,vy,x0,y0)
	fitLine(trace,line,cv::DIST_L2,0,0.01,0.01);

	traceModel.linesParam[stage][level].push_back(Vec4f(line));
}

void cv::softcascade::FastDtModel::computeModel(bool useTraceApp,bool useGeomHist){
	if(useTraceApp)
		traceModel.compute(paramDtFast.nScales);
	if(useGeomHist)
		geomModel.compute(imgSize, paramDtFast.nScales);
}

void cv::softcascade::FastDtModel::addStrongWithROI(Rect dw,double rank,uint level){

	geomModel.upperLeftPonts[level].push_back(GeomModel::StrongROI(dw,rank));

}
void cv::softcascade::FastDtModel::setGridsSize(std::vector<uint> grids){
	geomModel.gridsSize=grids;
}
std::vector<cv::softcascade::FastDtModel::Block>& cv::softcascade::FastDtModel::getBlocks4Grid(uint gridSize){
	return geomModel.grids[gridSize];
}
void cv::softcascade::FastDtModel::resolveWrongStd(){


	for(GeomModel::Grids::iterator itG=geomModel.grids.begin();itG!=geomModel.grids.end();++itG){
		for(std::vector<Block>::iterator itB=itG->second.begin();itB!=itG->second.end();++itB){
			for(std::vector<AverageCov>::iterator itC=itB->locationsHist.begin();itC!=itB->locationsHist.end();++itC){

//				std::cout<<"Average ("<<itC->avg.rows<<","<<itC->avg.cols<<")"<<std::endl;
//				std::cout<<"Cov ("<<itC->cov.rows<<","<<itC->cov.cols<<")"<<std::endl;

				// no information obtained to this level
				if(itC->avg.at<double>(0,0)==-1. && itC->avg.at<double>(0,1)==-1.)
					continue;

				if(itC->avg.at<double>(0,0)!=-1. && itC->avg.at<double>(0,1)!=-1. 	&&
						itC->cov.at<double>(0,0)==0. && itC->cov.at<double>(0,1)==0. &&
				        itC->cov.at<double>(1,0)==0. && itC->cov.at<double>(1,1)==0.){



					int maxW= std::min(std::abs(cvRound(itC->avg.at<double>(0,0)- itB->rect.x)),
										std::abs(cvRound(itC->avg.at<double>(0,0)-itB->rect.x-itB->rect.width))
									);

					int maxH=std::min( std::abs(cvRound(itC->avg.at<double>(0,1)- itB->rect.y)),
							std::abs(cvRound(itC->avg.at<double>(0,1)-itB->rect.y-itB->rect.height))
									);


					double K=4.61;
					itC->cov.at<double>(0,0)=(double)(pow(maxW,2)/(4.*K));
					itC->cov.at<double>(1,1)=(double)(pow(maxH,2)/(4.*K));
				}

				// eigen decomposition cov=VDV^(-1)
				Mat covVect=Mat(2,2,CV_64FC1);
				Mat covVal=Mat(2,1,CV_64FC1);
				eigen(itC->cov,covVal,covVect);

				// if cov is not definite positive (has at least one eigenvalue<=0)
				if(covVal.at<double>(0,0)<=0.|| covVal.at<double>(1,0)<=0.){
					if(covVal.at<double>(0,0)<=0.) covVal.at<double>(0,0)=0.2+DBL_EPSILON;
					if(covVal.at<double>(1,0)<=0.) covVal.at<double>(1,0)=0.2+DBL_EPSILON;
					itC->cov=covVect*Mat::diag(covVal)*covVect.inv();
					//itC->cov=covVect.inv()*Mat::diag(covVal)*covVect;
				}
			}
		}
	}
}
void cv::softcascade::FastDtModel::smoothLocations(){
	for(GeomModel::Grids::iterator itG=geomModel.grids.begin();itG!=geomModel.grids.end();++itG){
		for(std::vector<Block>::iterator itB=itG->second.begin();itB!=itG->second.end();++itB){
			// levelH point to levelHist vector data


			/*CvMat levelH= Mat(1,itB->levelsHist.size(),CV_64FC1,itB->levelsHist.data());

			cvSmooth(&levelH,&levelH, CV_GAUSSIAN, 3,0);*/
		}
	}
}

/*cv::softcascade::FastDtModel cv::softcascade::FastDtModel::operator+(const FastDtModel& model){


	FastDtModel modelSum(paramDtFast,dataset, numImages+model.numImages,imgSize);


	//------ Trace Approximation:  consider only modelL
	modelSum.traceModel=traceModel;

	//------ Geometry Model: merge both models

	//------ Copy the first gem. model
	modelSum.geomModel.grids=geomModel.grids;


	GeomModel::Grids::const_iterator itG=model.geomModel.grids.begin();

		for( ;itG!=model.geomModel.grids.end();++itG){


			double levHistAcc=std::accumulate(b->levelsHist.begin(),b->levelsHist.end(),0.);

						// energy
						b->energy=levHistAcc/energyTot[g->first];

						// levels histogram normalization
						if(levHistAcc>0){
							for(uint l=0;l<levels;l++)
								b->levelsHist[l]/=levHistAcc;
						}


			uint id=0;
			for(std::vector<Block>::const_iterator itB=itG->second.begin();itB!=itG->second.end();++itB){
				outFile<< (int)itG->first<<","<<id++;

				for(int i=0;i<itB->levelsHist.size();i++)
				outFile<< ","<< itB->levelsHist[i];


				for(std::vector<AverageCov>::const_iterator itA=itB->locationsHist.begin();itA!=itB->locationsHist.end();++itA)
					outFile<<","<< itA->avg.at<double>(0,0)<<","<<itA->avg.at<double>(1,0);

				for(std::vector<AverageCov>::const_iterator itC=itB->locationsHist.begin();itC!=itB->locationsHist.end();++itC)
					outFile<<","<< itC->cov.at<double>(0,0)<<","<<itC->cov.at<double>(0,1)<<","
					<<itC->cov.at<double>(1,0)<<","<<itC->cov.at<double>(1,1);


				outFile<<","<< itB->rect.x<<","<<itB->rect.y<<","<<itB->rect.width<<","<<itB->rect.height;
				outFile<<","<< itB->energy<<"\n";

			}
		}



	return modelSum;
}*/

void cv::softcascade::FastDtModel::saveModelIntoDat(String path, String prefix){

	// Trace Approximation
	std::cout<<"Saving Trace Approximation...";
	std::ofstream outFile;
	outFile.open((path+"/"+prefix+"_"+TraceModel::TRACEMODEL+".dat").c_str(),std::ofstream::out);

	TraceModel::SlopesMap::const_iterator itS= traceModel.slopes.begin();

	for( ;itS!=traceModel.slopes.end();++itS){
		outFile << (int)itS->first;

		for(std::vector<double >::const_iterator itL=itS->second.begin();itL!=itS->second.end();++itL){
			outFile << ",";
			outFile<< *itL;
		}
		outFile << "\n";
	}
	outFile.close();
	std::cout<<"\tCOMPLETED"<<std::endl;

	// Geometry Model
	std::cout<<"Saving Geometric Histograms...";
	outFile.open((path+"/"+prefix+"_"+GeomModel::GEOMMODEL+".dat").c_str(),std::ofstream::out);

	GeomModel::Grids::const_iterator itG=geomModel.grids.begin();

	for( ;itG!=geomModel.grids.end();++itG){

		uint id=0;
		for(std::vector<Block>::const_iterator itB=itG->second.begin();itB!=itG->second.end();++itB){
			outFile<< (int)itG->first<<","<<id++;

			for(uint i=0;i<itB->levelsHist.size();i++)
				outFile<< ","<< itB->levelsHist[i];

			for(std::vector<AverageCov>::const_iterator itA=itB->locationsHist.begin();itA!=itB->locationsHist.end();++itA)
				outFile<<","<< itA->avg.at<double>(0)<<","<<itA->avg.at<double>(1);

			for(std::vector<AverageCov>::const_iterator itC=itB->locationsHist.begin();itC!=itB->locationsHist.end();++itC)
				outFile<<","<< itC->cov.at<double>(0,0)<<","<<itC->cov.at<double>(0,1)<<","
				<<itC->cov.at<double>(1,0)<<","<<itC->cov.at<double>(1,1);


			outFile<<","<< itB->rect.x<<","<<itB->rect.y<<","<<itB->rect.width<<","<<itB->rect.height;
			outFile<<","<< itB->energy<<"\n";
		}
	}
	std::cout<<"\tCOMPLETED";
}

cv::softcascade::DetectorFast::DetectorFast(ParamDetectorFast param)
:Detector(param.minScale, param.maxScale, param.nScales, param.nMS),fastModel(param)
{

}

cv::softcascade::DetectorFast::~DetectorFast() {}

bool cv::softcascade::DetectorFast::loadModel(const FileNode& fastNode){

	// save recjection criteria and octave'paramters for restore it later
	tempI.rejCriteria=rejCriteria;

	tempI.index=new int[fields->octaves.size()];
	tempI.weaks=new int[fields->octaves.size()];

	for(uint i=0;i<fields->octaves.size();i++){
		tempI.index[i]=fields->octaves[i].index;
		tempI.weaks[i]=fields->octaves[i].weaks;
	}

	try{
		fastNode >> fastModel;
	}catch (Exception& e) {
		return false;
	}

	// Geometry model: if covariance=0 than set to block size
	//fastModel.resolveWrongStd();

	// Geometry model: smooth the  levelHist energy (Gaussian kernel of 1x3 )
	//fastModel.smoothLocations();

	return true;
}

//------------ Random sampling functions ------------------
double *cv::softcascade::DetectorFast::r8vec_uniform_01_new ( int n, int *seed )
{
  int i;
  int i4_huge = 2147483647;
  int k;
  double *r;

  if ( *seed == 0 )
  {
    std::cerr << "\n";
    std::cerr << "R8VEC_UNIFORM_01_NEW - Fatal error!\n";
    std::cerr << "  Input value of SEED = 0.\n";
    exit ( 1 );
  }

  r = new double[n];

  for ( i = 0; i < n; i++ )
  {
    k = *seed / 127773;

    *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

    if ( *seed < 0 )
    {
      *seed = *seed + i4_huge;
    }

    r[i] = ( double ) ( *seed ) * 4.656612875E-10;
  }

  return r;
}

double *cv::softcascade::DetectorFast::r8po_fa ( int n, double a[] )
{
  double *b;
  int i;
  int j;
  int k;
  double s;

  b = new double[n*n];

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < n; i++ )
    {
      b[i+j*n] = a[i+j*n];
    }
  }

  for ( j = 0; j < n; j++ )
  {
    for ( k = 0; k <= j-1; k++ )
    {
      for ( i = 0; i <= k-1; i++ )
      {
        b[k+j*n] = b[k+j*n] - b[i+k*n] * b[i+j*n];
      }
      b[k+j*n] = b[k+j*n] / b[k+k*n];
    }

    s = b[j+j*n];
    for ( i = 0; i <= j-1; i++ )
    {
      s = s - b[i+j*n] * b[i+j*n];
    }

    if ( s <= 0.0 )
    {
      delete [] b;
      return NULL;
    }

    b[j+j*n] = sqrt ( s );
  }

  //
//  Since the Cholesky factor is in R8GE format, zero out the lower triangle.
//
  for ( i = 0; i < n; i++ )
  {
    for ( j = 0; j < i; j++ )
    {
      b[i+j*n] = 0.0;
    }
  }

  return b;
}

double *cv::softcascade::DetectorFast::r8vec_normal_01_new ( int n, int *seed )
{
# define PI 3.141592653589793

  int i;
  int m;
  static int made = 0;
  double *r;
  static int saved = 0;
  double *x;
  int x_hi;
  int x_lo;
  static double y = 0.0;

  x = new double[n];
//
//  I'd like to allow the user to reset the internal data.
//  But this won't work properly if we have a saved value Y.
//  I'm making a crock option that allows the user to signal
//  explicitly that any internal memory should be flushed,
//  by passing in a negative value for N.
//
  if ( n < 0 )
  {
    made = 0;
    saved = 0;
    y = 0.0;
    return NULL;
  }
  else if ( n == 0 )
  {
    return NULL;
  }
//
//  Record the range of X we need to fill in.
//
  x_lo = 1;
  x_hi = n;
//
//  Use up the old value, if we have it.
//
  if ( saved == 1 )
  {
    x[0] = y;
    saved = 0;
    x_lo = 2;
  }
//
//  Maybe we don't need any more values.
//
  if ( x_hi - x_lo + 1 == 0 )
  {
  }
//
//  If we need just one new value, do that here to avoid null arrays.
//
  else if ( x_hi - x_lo + 1 == 1 )
  {
    r = r8vec_uniform_01_new ( 2, seed );

    x[x_hi-1] = sqrt ( - 2.0 * log ( r[0] ) ) * cos ( 2.0 * PI * r[1] );
    y =         sqrt ( - 2.0 * log ( r[0] ) ) * sin ( 2.0 * PI * r[1] );

    saved = 1;

    made = made + 2;

    delete [] r;
  }
//
//  If we require an even number of values, that's easy.
//
  else if ( ( x_hi - x_lo + 1 ) % 2 == 0 )
  {
    m = ( x_hi - x_lo + 1 ) / 2;

    r = r8vec_uniform_01_new ( 2*m, seed );

    for ( i = 0; i <= 2*m-2; i = i + 2 )
    {
      x[x_lo+i-1] = sqrt ( - 2.0 * log ( r[i] ) ) * cos ( 2.0 * PI * r[i+1] );
      x[x_lo+i  ] = sqrt ( - 2.0 * log ( r[i] ) ) * sin ( 2.0 * PI * r[i+1] );
    }
    made = made + x_hi - x_lo + 1;

    delete [] r;
  }
//
//  If we require an odd number of values, we generate an even number,
//  and handle the last pair specially, storing one in X(N), and
//  saving the other for later.
//
  else
  {
    x_hi = x_hi - 1;

    m = ( x_hi - x_lo + 1 ) / 2 + 1;

    r = r8vec_uniform_01_new ( 2*m, seed );

    for ( i = 0; i <= 2*m-4; i = i + 2 )
    {
      x[x_lo+i-1] = sqrt ( - 2.0 * log ( r[i] ) ) * cos ( 2.0 * PI * r[i+1] );
      x[x_lo+i  ] = sqrt ( - 2.0 * log ( r[i] ) ) * sin ( 2.0 * PI * r[i+1] );
    }

    i = 2*m - 2;

    x[x_lo+i-1] = sqrt ( - 2.0 * log ( r[i] ) ) * cos ( 2.0 * PI * r[i+1] );
    y           = sqrt ( - 2.0 * log ( r[i] ) ) * sin ( 2.0 * PI * r[i+1] );

    saved = 1;

    made = made + x_hi - x_lo + 2;

    delete [] r;
  }

  return x;
}

double *cv::softcascade::DetectorFast::multinormal_sample( int m, int n, double a[], double mu[], int *seed )
{
  int i;
  int j;
  int k;
  double *r;
  double *x;
  double *y;
//
//  Compute the upper triangular Cholesky factor R of the variance-covariance
//  matrix.
//
  r = r8po_fa( m, a );

  if ( !r )
  {
    std::cout << "\n";
    std::cout << "MULTINORMAL_SAMPLE - Fatal error!\n";
    std::cout << "  The variance-covariance matrix is not positive definite symmetric\n";
    exit ( 1 );
  }
//
//  Y = MxN matrix of samples of the 1D normal distribution with mean 0
//  and variance 1.
//
  y = r8vec_normal_01_new ( m*n, seed );
//
//  Compute X = MU + R' * Y.
//
  x = new double[m*n];

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      x[i+j*m] = mu[i];
      for ( k = 0; k < m; k++ )
      {
        x[i+j*m] = x[i+j*m] + r[k+i*m] * y[k+j*m];
      }
    }
  }

  delete [] r;
  delete [] y;

  return x;
}

//---------------------------------------------------------

bool cv::softcascade::DetectorFast::load(const FileNode& cascadeModel)
{
	return Detector::load(cascadeModel);
}


bool cv::softcascade::DetectorFast::load(const FileNode& cascadeModel,const FileNode& ftModel)
{
	return Detector::load(cascadeModel)&&loadModel(ftModel);
}
void cv::softcascade::DetectorFast::saveModelIntoDat(String path){
	fastModel.saveModelIntoDat(path);
}

void cv::softcascade::DetectorFast::detectFastWithMask(cv::InputArray _image, cv::InputArray _m, double th,std::vector<Detection>& objects){
	/*
		uint lastStage=fastModel.paramDtFast.lastStage;
		// set new recjection criteria and octave'paramters for tracerestore it later
		rejCriteria=NO_REJECT;
		for (uint i=0;i<fields->octaves.size();i++){
			fields->octaves[i].index=(int)(fields->octaves[i].weaks*fields->octaves[i].index)/lastStage;
			fields->octaves[i].weaks= lastStage;
		}


		double slope;
	*/
		uint currentSize;

		std::vector<FastDtModel::Block> blocks=fastModel.getBlocks4Grid(fastModel.paramDtFast.gridSize);

	    // only color images are suppered
	    cv::Mat image = _image.getMat();
	    CV_Assert(image.type() == CV_8UC3);
		cv::Mat mask= _m.getMat();
		mask.convertTo(mask, CV_32SC1);


	    fields->calcLevels(image.size(),(float) minScale, (float)maxScale, scales);
	    objects.clear();

	//--- Execution detection phase with NO_REJECT (detectNoROI function) -

		Fields& fld = *fields;
	    // create integrals
	   ChannelStorage storage(image, fld.shrinkage, fld.featureTypeStr);

	    typedef std::vector<Level>::const_iterator lIt;

	    // compute the pyramid size (total number of dw)
	    double pyramidSize=0.;
	    for (lIt it = fld.levels.begin(); it != fld.levels.end(); ++it)
	    {
	        const Level& level = *it;

	        // we train only 3 scales.
	        if (level.origScale > 2.5) break;

	        pyramidSize+=(double)level.workRect.height*level.workRect.width;

	    }
	//    std::cout<< "Size of Pyramid: "<<pyramidSize<< "dw"<< std::endl;

	    for (lIt it = fld.levels.begin(); it != fld.levels.end(); ++it)
	    {
	        const Level& level = *it;

	        // we train only 3 scales.
	        if (level.origScale > 2.5) break;

	        currentSize=objects.size();

	        //############ Sampling by geometric model #######################
	//  std::cout<<std::endl<<"\t level "<<it - fld.levels.begin()<<std::endl;

	        for(uint b=0;b<blocks.size();b++){

	//        	std::cout<<"\tBlock: "<<b;
	//        	std::cout<<"\t Level "<<it-fld.levels.begin()<<"["<<level.workRect.width<<","<<level.workRect.height<<"]"<<std::endl;

	        	int samplesTot=cvRound(fastModel.paramDtFast.gamma*pyramidSize*blocks[b].energy*blocks[b].levelsHist[it-fld.levels.begin()]);

	//        	std::cout<< " n° dw to extract: "<< remainingSamp<<std::endl;
	        	if(samplesTot==0)
	        		continue;

				double motion_sum;

	        	//Uniform sampling ( avg=(-1,-1) )
	        	if( isUniformSampling(blocks[b].locationsHist[it-fld.levels.begin()].avg) ){

	//        		std::cout<<"\t Uniform Sampling ("<<samplesTot<<")"<<std::endl;

	        		int startX,startY, endX, endY, stepX, stepY;
	        		uniSamples(blocks[b],level,startX,startY, endX, endY, stepX, stepY,samplesTot);


	                for (int dy = startY; dy < endY; dy+=stepY)
	                {
	                    for (int dx = startX; dx < endX; dx+=stepX)
	                    {
							motion_sum=0.;

							for(int y=cvRound(dy * fld.shrinkage); y<cvRound(dy * fld.shrinkage)+level.objSize.height;y++){
								for(int x=cvRound(dx * fld.shrinkage); x<=cvRound(dx * fld.shrinkage)+level.objSize.width;x++){
									motion_sum+= mask.ptr<int>(y)[x];
								}
							}

							if(motion_sum<th*(level.objSize.width*level.objSize.height))
									continue;

	                    	storage.offset = (int)(dy * storage.step + dx);
	                        fld.detectAt(dx, dy, level, storage, objects);
	                    }
	                }
	        	}
	        	// Normal Sampling
	        	else{
	/*
		      	std::cout<<"\t Random Sampling ("<<samplesTot<<") ";
					std::cout<<"Average("<< blocks[b].locationsHist[it-fld.levels.begin()].avg.at<double>(0,0)<<","
											<<blocks[b].locationsHist[it-fld.levels.begin()].avg.at<double>(0,1)<<") ";

					std::cout<<"Cov ("<< blocks[b].locationsHist[it-fld.levels.begin()].cov.at<double>(0,0)<<","
											<<blocks[b].locationsHist[it-fld.levels.begin()].cov.at<double>(0,1)<<","
											<<blocks[b].locationsHist[it-fld.levels.begin()].cov.at<double>(1,0)<<","
											<<blocks[b].locationsHist[it-fld.levels.begin()].cov.at<double>(1,1)<<")"<<std::endl;
	*/
					std::set<Point2i,classPoint2iComp> dw;

					cv::Mat covM; blocks[b].locationsHist[it-fld.levels.begin()].cov.copyTo(covM);
					cv::Mat avgM=blocks[b].locationsHist[it-fld.levels.begin()].avg;

					rndSamples(level,dw,avgM,covM,samplesTot);


					for (std::set<Point2i >::iterator itDW=dw.begin();itDW!=dw.end();++itDW){
						motion_sum=0.;

						for(int y=cvRound(itDW->y* fld.shrinkage); y<cvRound(itDW->y* fld.shrinkage)+level.objSize.height;y++){
							for(int x=cvRound(itDW->x* fld.shrinkage); x<=cvRound(itDW->x* fld.shrinkage)+level.objSize.width;x++){
								motion_sum+= mask.ptr<int>(y)[x];
							}
						}
						if(motion_sum<th*(level.objSize.width*level.objSize.height))
								continue;

						storage.offset = itDW->y * storage.step + itDW->x;
						fld.detectAt(itDW->x, itDW->y, level, storage, objects);
					}
				}

		/*
				//############# Compute the final score for each positive dw by approximation ###########
				fastModel.getSlopeAt(lastStage,it-fld.levels.begin(),slope);

				for(uint i=currentSize;i<objects.size();i++){
					objects[i].confidence+=(1024-lastStage)*slope;
				}
				//######################################################################
		  */

			}
	   	}




	/*
	//-------------------------------------------------------------------

	//----------- Restore recjection criteria and octave'paramters ------
		rejCriteria=tempI.rejCriteria;
		for (uint i=0;i<fields->octaves.size();i++){
			fields->octaves[i].index=tempI.index[i];
			fields->octaves[i].weaks= tempI.weaks[i];
		}
	//--------------------------------------------------------------------
	*/


		if (rejCriteria != NO_REJECT) suppress(rejCriteria, objects);


}

void cv::softcascade::DetectorFast::uniSamples(const FastDtModel::Block& block, const Level& level,
		int& startX,int& startY, int& endX, int& endY, int& stepX, int& stepY,
		const int samplesTot)
{

	startX = cvRound((double)(block.rect.x)/fields->shrinkage);
	startY = cvRound((double)(block.rect.y)/fields->shrinkage);
	endX   = std::min(cvRound((double)(block.rect.x+block.rect.width) /fields->shrinkage),
						  level.workRect.width-1);
	endY   = std::min(cvRound((double)(block.rect.y+block.rect.height) /fields->shrinkage),
						  level.workRect.height-1);

	stepX=cvRound((double)(endX-startX+1)/std::sqrt(samplesTot));
	if(stepX<1)
		stepX=1;

	stepY=cvRound((double)(endY-startY+1)/std::sqrt(samplesTot));
	if(stepY<1)
		stepY=1;
}
void cv::softcascade::DetectorFast::rndSamples(const Level& level,
		std::set<Point2i,classPoint2iComp>& dw, cv::Mat& avgM, cv::Mat& covM,const int samplesTot){

	for(uint round=0;round<fastModel.paramDtFast.round;round++){

//					std::cout<< "--- Round "<<round<<" ---";

		// sampling
		int seed=time(NULL);
		int samplesR=samplesTot-(int)(dw.size());

		double* sampling= multinormal_sample(2, samplesR, covM.ptr<double>(0),
				avgM.ptr<double>(0), &seed);
		// print generated sampling
//					std::cout<<" samples: "<<std::endl;
		for(int i=0; i<samplesR;i++){
//					std::cout<<"["<<level.workRect.width<<","<<level.workRect.height<<"]-"<<round<<"\t("<<sampling[2*i]<< " , "<< sampling[2*i+1]<< ") --->";

			// Rescale upperLeftpoints by shrinkage and check the validity
			sampling[2*i]=  (sampling[2*i])/(fields->shrinkage);
			sampling[2*i+1]=(sampling[2*i+1])/(fields->shrinkage);
//						std::cout<<" ("<<sampling[2*i]<< " , "<< sampling[2*i+1]<< ") ";

			int samp_X=cvRound(sampling[2*i]);
			int samp_Y=cvRound(sampling[2*i+1]);


			if(samp_X>=0 && samp_Y>=0 && samp_X< level.workRect.width && samp_Y<level.workRect.height){
				dw.insert(Point2i(samp_X,samp_Y));
//		        			std::cout<<"VALID: ("<<samp_X<< ","<<samp_Y<<")"<<std::endl;
			}
/*						else{
				std::cout<<"<<< NOT VALID!! >>>"<<std::endl;
			}
*/
		}

		delete[] sampling;

//					std::cout<<std::endl;
//					std::cout<< "\t \t Valid sampled extracted so far: "<<dw.size()<<" of "<<samplesTot<<std::endl;

		if(samplesR==0)
			break;

		if(fastModel.paramDtFast.covMExpansion){
			covM*=((1.5)*(1.5));

		}
	}


}
void cv::softcascade::DetectorFast::rndSamples_3D(std::vector<Level>::const_iterator lIt,std::vector<Level>::const_iterator endIt,
		std::set<Point3i,classPoint3iComp>& dw, cv::Mat& avgM, cv::Mat& covM,const int samplesTot){

	for(uint round=0;round<fastModel.paramDtFast.round;round++){

//					std::cout<< "--- Round "<<round<<" ---";

		// sampling
		int seed=time(NULL);
		int samplesR=samplesTot-(int)(dw.size());

		double* sampling= multinormal_sample(3, samplesR, covM.ptr<double>(0),
				avgM.ptr<double>(0), &seed);
		// print generated sampling
//					std::cout<<" samples: "<<std::endl;
		for(int i=0; i<samplesR;i++){
//					std::cout<<"["<<level.workRect.width<<","<<level.workRect.height<<"]-"<<round<<"\t("<<sampling[2*i]<< " , "<< sampling[2*i+1]<< ") --->";

			// Rescale upperLeftpoints by shrinkage and check the validity
			sampling[3*i]=  (sampling[3*i])/(fields->shrinkage);
			sampling[3*i+1]=(sampling[3*i+1])/(fields->shrinkage);

//						std::cout<<" ("<<sampling[2*i]<< " , "<< sampling[2*i+1]<< ") ";

			int samp_X=cvRound(sampling[3*i]);
			int samp_Y=cvRound(sampling[3*i+1]);
			int samp_L=cvRound(sampling[3*i+2]);

			if (lIt+samp_L >=endIt)
				continue;

			const Level& level = *(lIt+samp_L);

			if(samp_X>=0 && samp_Y>=0 && samp_X< level.workRect.width && samp_Y<level.workRect.height){
				dw.insert(Point3i(samp_X,samp_Y,samp_L));
			}
		}

		delete[] sampling;

		if(samplesR==0)
			break;

		if(fastModel.paramDtFast.covMExpansion){
			covM*=((1.5)*(1.5));

		}
	}


}


void cv::softcascade::DetectorFast::generateSamples(cv::Size imgSize,std::vector<Sample>& samples){
		uint currentSize;

		std::vector<FastDtModel::Block> blocks=fastModel.getBlocks4Grid(fastModel.paramDtFast.gridSize);


	    fields->calcLevels(imgSize,(float) minScale, (float)maxScale, scales);
	    samples.clear();

	//--- Execution detection phase with NO_REJECT (detectNoROI function) -

		Fields& fld = *fields;

	    typedef std::vector<Level>::const_iterator lIt;

	    // compute the pyramid size (total number of dw)
	    double pyramidSize=0.;
	    for (lIt it = fld.levels.begin(); it != fld.levels.end(); ++it)
	    {
	        const Level& level = *it;

	        // we train only 3 scales.
	        if (level.origScale > 2.5) break;

	        pyramidSize+=(double)level.workRect.height*level.workRect.width;

	    }
		std::cout<<"Grid: "<<fastModel.paramDtFast.gridSize<<" - Gamma:"<<fastModel.paramDtFast.gamma;
	    std::cout<< " Size of Pyramid: "<<pyramidSize<< "dw"<< std::endl;

		double uEnergy=1./(fastModel.paramDtFast.gridSize*fastModel.paramDtFast.gridSize);

	    for (lIt it = fld.levels.begin(); it != fld.levels.end(); ++it)
	    {
	        const Level& level = *it;

	        // we train only 3 scales.
	        if (level.origScale > 2.5) break;

	        currentSize=samples.size();

	        //############ Sampling by geometric model #######################

	        std::cout<< "\tLevel "<<it-fld.levels.begin()<< ":"<< std::endl;
	        for(uint b=0;b<blocks.size();b++){

	        	std::cout<< "\t\tBlock "<<b<<" ["<<blocks[b].rect.x<<","<<blocks[b].rect.y
	        			<<","<<blocks[b].rect.width<<","<<blocks[b].rect.height<<"]";

	        	int samplesTot;

	        	if(fastModel.paramDtFast.uniformEnergy){
	        		samplesTot=cvRound(fastModel.paramDtFast.gamma*pyramidSize*uEnergy*blocks[b].levelsHist[it-fld.levels.begin()]);
	        	}
	        	else
	        		samplesTot=cvRound(fastModel.paramDtFast.gamma*pyramidSize*blocks[b].energy*blocks[b].levelsHist[it-fld.levels.begin()]);

	        	if(samplesTot==0){
	        		std::cout<< " EMPTY\n";
	        		continue;
	        	}
	        	std::cout<<"["<<samplesTot<<"] ";
	        	//Uniform sampling ( avg=(-1,-1) )
	        	if( isUniformSampling(blocks[b].locationsHist[it-fld.levels.begin()].avg) ){
	        		std::cout<< " Uniform[";

	        		int startX,startY, endX, endY, stepX, stepY;
	        		uniSamples(blocks[b],level,startX,startY, endX, endY, stepX, stepY,samplesTot);

	        		int numS=0;
	        		std::cout<<startX<< ":"<<stepX<<":"<<endX<<"] ["<<startY<<":"<<stepY<<":"<<endY<<"] ";
	                for (int dy = startY; dy < endY; dy+=stepY)
	                {
	                    for (int dx = startX; dx < endX; dx+=stepX)
	                    {
	                    	Rect rect((dx * fld.shrinkage),(dy * fld.shrinkage),
	                    			level.objSize.width,level.objSize.height);

	                    	samples.push_back(Sample(rect, it - fld.levels.begin(),Sample::UNIFORM));
	                    	numS++;
	                    }
	                }
	                std::cout<< "("<<numS<<") - Workrect["<<level.workRect.width<<","<<level.workRect.height<<"]"<<std::endl;
	        	}
	        	// Normal Sampling
	        	else{

					std::set<Point2i,classPoint2iComp> dw;


					cv::Mat covM(2,2,CV_64FC1);blocks[b].locationsHist[it-fld.levels.begin()].cov.copyTo(covM);
					cv::Mat avgM=blocks[b].locationsHist[it-fld.levels.begin()].avg;

					rndSamples(level,dw,avgM,covM,samplesTot);


					for (std::set<Point2i >::iterator itDW=dw.begin();itDW!=dw.end();++itDW){

                    	Rect rect(itDW->x* fld.shrinkage,itDW->y* fld.shrinkage,
                    			level.objSize.width,level.objSize.height);

						samples.push_back(Sample(rect, it - fld.levels.begin(),Sample::NORMAL));
					}
					std::cout<< " Normal("<<dw.size()<<")"<<std::endl;
				}

			}
	   	}
}

void cv::softcascade::DetectorFast::detectFast(cv::InputArray _image,std::vector<Detection>& objects)
{
/*
	uint lastStage=fastModel.paramDtFast.lastStage;
	// set new recjection criteria and octave'paramters for tracerestore it later
	rejCriteria=NO_REJECT;
	for (uint i=0;i<fields->octaves.size();i++){
		fields->octaves[i].index=(int)(fields->octaves[i].weaks*fields->octaves[i].index)/lastStage;
		fields->octaves[i].weaks= lastStage;
	}


	double slope;
*/


    // only color images are suppered
    cv::Mat image = _image.getMat();
    CV_Assert(image.type() == CV_8UC3);


    fields->calcLevels(image.size(),(float) minScale, (float)maxScale, scales);
    objects.clear();

//--- Execution detection phase with NO_REJECT (detectNoROI function) -

	Fields& fld = *fields;
    // create integrals
   ChannelStorage storage(image, fld.shrinkage, fld.featureTypeStr);

    typedef std::vector<Level>::const_iterator lIt;

    // compute the pyramid size (total number of dw)
    double pyramidSize=0.;
    for (lIt it = fld.levels.begin(); it != fld.levels.end(); ++it)
    {
        const Level& level = *it;

        // we train only 3 scales.
        if (level.origScale > 2.5) break;

        pyramidSize+=(double)level.workRect.height*level.workRect.width;
    }

//    std::cout<< "Size of Pyramid: "<<pyramidSize<< "dw"<< std::endl;


    switch (fastModel.paramDtFast.geomModelType) {

//########################### FIXED_BLOCK ######################
    	case 0:
    		detectFast_FIXED_GRID(objects,storage,fld,pyramidSize);
			break;

//########################### GMM ##############################
    	case 1:
    		detectFast_GMM(objects,storage,fld,pyramidSize);
    		break;
//########################### GMM 3D ##########################
    	case 2:
    	    detectFast_GMM_3D(objects,storage,fld,pyramidSize);
    		break;
//########################### SEGMENTATION ##########################
    	case 3:
    		detectFast_SEG(objects,storage,fld,pyramidSize);
    		break;

    	default:
    		std::cerr<<"Type of Geometric Model unknow";
			break;
	}




/*
//-------------------------------------------------------------------

//----------- Restore recjection criteria and octave'paramters ------
	rejCriteria=tempI.rejCriteria;
	for (uint i=0;i<fields->octaves.size();i++){
		fields->octaves[i].index=tempI.index[i];
		fields->octaves[i].weaks= tempI.weaks[i];
	}
//--------------------------------------------------------------------
*/


	if (rejCriteria != NO_REJECT) suppress(rejCriteria, objects);

}
void cv::softcascade::DetectorFast::detectFast_FIXED_GRID(std::vector<Detection>& objects, ChannelStorage& storage, Fields& fld,double pyramidSize){
    typedef std::vector<Level>::const_iterator lIt;

    std::vector<FastDtModel::Block> blocks=fastModel.getBlocks4Grid(fastModel.paramDtFast.gridSize);

	double uEnergy=1./(fastModel.paramDtFast.gridSize*fastModel.paramDtFast.gridSize);

    for (lIt it = fld.levels.begin(); it != fld.levels.end(); ++it)
    {
        const Level& level = *it;

        // we train only 3 scales.
        if (level.origScale > 2.5) break;


        //############ Sampling by geometric model #######################
//  std::cout<<std::endl<<"\t level "<<it - fld.levels.begin()<<std::endl;

        for(uint b=0;b<blocks.size();b++){

//        	std::cout<<"\tBlock: "<<b;
//        	std::cout<<"\t Level "<<it-fld.levels.begin()<<"["<<level.workRect.width<<","<<level.workRect.height<<"]"<<std::endl;

        	int samplesTot;

        	if(fastModel.paramDtFast.uniformEnergy){
        		samplesTot=cvRound(fastModel.paramDtFast.gamma*pyramidSize*uEnergy*blocks[b].levelsHist[it-fld.levels.begin()]);
        	}
        	else
        		samplesTot=cvRound(fastModel.paramDtFast.gamma*pyramidSize*blocks[b].energy*blocks[b].levelsHist[it-fld.levels.begin()]);

        	//        	std::cout<< " n° dw to extract: "<< remainingSamp<<std::endl;
        	if(samplesTot==0)
        		continue;

        	//Uniform sampling ( avg=(-1,-1) )
        	if( isUniformSampling(blocks[b].locationsHist[it-fld.levels.begin()].avg) ){
//        		std::cout<<"\t Uniform Sampling ("<<samplesTot<<")"<<std::endl;
        		int startX,startY, endX, endY, stepX, stepY;
        		uniSamples(blocks[b],level,startX,startY, endX, endY, stepX, stepY,samplesTot);


                for (int dy = startY; dy < endY; dy+=stepY)
                {
                    for (int dx = startX; dx < endX; dx+=stepX)
                    {
                    	storage.offset = (int)(dy * storage.step + dx);
                        fld.detectAt(dx, dy, level, storage, objects);
                    }
                }
        	}
        	// Normal Sampling
        	else{
				std::set<Point2i,classPoint2iComp> dw;


				cv::Mat covM(2,2,CV_64FC1);blocks[b].locationsHist[it-fld.levels.begin()].cov.copyTo(covM);
				cv::Mat avgM=blocks[b].locationsHist[it-fld.levels.begin()].avg;

				rndSamples(level,dw,avgM,covM,samplesTot);


				for (std::set<Point2i >::iterator itDW=dw.begin();itDW!=dw.end();++itDW){
					storage.offset = itDW->y * storage.step + itDW->x;
					fld.detectAt(itDW->x, itDW->y, level, storage, objects);
				}
			}
		}
   	}
}
void cv::softcascade::DetectorFast::detectFast_GMM(std::vector<Detection>& objects, ChannelStorage& storage, Fields& fld,double pyramidSize){
    typedef std::vector<Level>::const_iterator lIt;



	double uEnergy=0.;
	for (lIt it = fld.levels.begin(); it != fld.levels.end(); ++it)
		uEnergy+=(double)(fastModel.getMixtureCAtLevel(it -fld.levels.begin()).avgCov.size());

	uEnergy=1./uEnergy;

    for (lIt it = fld.levels.begin(); it != fld.levels.end(); ++it)
    {
        const Level& level = *it;

        // we train only 3 scales.
        if (level.origScale > 2.5) break;

        uint levelId=it -fld.levels.begin();



        FastDtModel::MixtureC mixC=fastModel.getMixtureCAtLevel(levelId);

        for(uint cI=0;cI<mixC.avgCov.size();cI++){


        	int samplesTot;

        	if(fastModel.paramDtFast.uniformEnergy){
        		samplesTot=cvRound(fastModel.paramDtFast.gamma*pyramidSize*uEnergy*mixC.energy[cI]);
        	}
        	else
        		samplesTot=cvRound(fastModel.paramDtFast.gamma*pyramidSize*mixC.energy[cI]);

        	//        	std::cout<< " n° dw to extract: "<< remainingSamp<<std::endl;
        	if(samplesTot==0)
        		continue;


        	// Normal Sampling
			std::set<Point2i,classPoint2iComp> dw;


			cv::Mat covM(2,2,CV_64FC1);mixC.avgCov[cI].cov.copyTo(covM);
			cv::Mat avgM=mixC.avgCov[cI].avg;

			rndSamples(level,dw,avgM,covM,samplesTot);


			for (std::set<Point2i >::iterator itDW=dw.begin();itDW!=dw.end();++itDW){
				storage.offset = itDW->y * storage.step + itDW->x;
				fld.detectAt(itDW->x, itDW->y, level, storage, objects);
			}

		}
   	}
}
void cv::softcascade::DetectorFast::detectFast_SEG(std::vector<Detection>& objects, ChannelStorage& storage, Fields& fld,double pyramidSize){
    typedef std::vector<Level>::const_iterator lIt;



    for (lIt it = fld.levels.begin(); it != fld.levels.end(); ++it)
    {
        const Level& level = *it;

        // we train only 3 scales.
        if (level.origScale > 2.5) break;

        uint levelId=it -fld.levels.begin();


        // Normal Sampling
		std::set<Point2i,classPoint2iComp>& dw= fastModel.getUpperLeftPointsAtLevel(levelId);

		for (std::set<Point2i >::iterator itDW=dw.begin();itDW!=dw.end();++itDW){
			storage.offset = itDW->y * storage.step + itDW->x;
			fld.detectAt(itDW->x, itDW->y, level, storage, objects);
		}

   	}
}



void cv::softcascade::DetectorFast::detectFast_GMM_3D(std::vector<Detection>& objects, ChannelStorage& storage, Fields& fld,double pyramidSize){
    typedef std::vector<Level>::const_iterator lIt;



	FastDtModel::MixtureC mixC=fastModel.getMixtureCAtLevel(0);

	double uEnergy=1./(mixC.avgCov.size());

	for(uint cI=0;cI<mixC.avgCov.size();cI++){
		int samplesTot;

		if(fastModel.paramDtFast.uniformEnergy){
			samplesTot=cvRound(fastModel.paramDtFast.gamma*pyramidSize*uEnergy*mixC.energy[cI]);
		}
		else
			samplesTot=cvRound(fastModel.paramDtFast.gamma*pyramidSize*mixC.energy[cI]);

		//        	std::cout<< " n° dw to extract: "<< remainingSamp<<std::endl;
		if(samplesTot==0)
			continue;


		// Normal Sampling
		std::set<Point3i,classPoint3iComp> dw;


		cv::Mat covM(3,3,CV_64FC1);mixC.avgCov[cI].cov.copyTo(covM);
		cv::Mat avgM=mixC.avgCov[cI].avg;

		rndSamples_3D(fld.levels.begin(),fld.levels.end(),dw,avgM,covM,samplesTot);

		for (std::set<Point3i >::iterator itDW=dw.begin();itDW!=dw.end();++itDW){
			const Level& level = *(fld.levels.begin()+itDW->z);
			if (level.origScale > 2.5) continue;

			storage.offset = itDW->y * storage.step + itDW->x;
			fld.detectAt(itDW->x, itDW->y, level, storage, objects);
		}

	}
}


inline uint cv::softcascade::DetectorFast::getNumLevels(){
	return fields->levels.size();
}

// ============================================================================================================== //
//		     Implementation of DetectorTrace (without trace evaluation reduction) and other structures nedded
// ============================================================================================================= //

struct ConfidenceGtTrace
{
    bool operator()(const cv::softcascade::Trace& a, const cv::softcascade::Trace& b) const
    {
        return a.detection.confidence > b.detection.confidence;
    }
};
// detect local maximum, maintaining the rest of traces
void DollarNMSTrace(std::vector<cv::softcascade::Trace>& positiveTrace, bool noMaxSupp)
{

	std::vector<cv::softcascade::Trace> objects=positiveTrace;

    static const float DollarThreshold = 0.65f;
    std::sort(objects.begin(), objects.end(), ConfidenceGtTrace());

    for (std::vector<cv::softcascade::Trace>::iterator dIt = objects.begin(); dIt != objects.end(); ++dIt)
    {
        const Detection &a = dIt->detection;
        positiveTrace[dIt->index].classType=cv::softcascade::Trace::LOCALMAXIMUM;
        positiveTrace[dIt->index].localMaxIndex=dIt->index;

        uint64 lowScoreI=dIt->index;
        for (std::vector<cv::softcascade::Trace>::iterator next = dIt + 1; next != objects.end(); )
        {
            const Detection &b = next->detection;

            const float ovl =  overlap(a.bb(), b.bb()) / std::min(a.bb().area(), b.bb().area());

            if (ovl > DollarThreshold){
            	positiveTrace[next->index].localMaxIndex=dIt->index;
            	lowScoreI=next->index;
            	next = objects.erase(next);
            }
            else
                ++next;
        }
        // set lowest trace Index
        positiveTrace[dIt->index].localMinIndex=lowScoreI;

    }
    if(noMaxSupp){
        for (std::vector<cv::softcascade::Trace>::iterator it = positiveTrace.begin(); it != positiveTrace.end();){
        	if(it->classType!=cv::softcascade::Trace::LOCALMAXIMUM)
        		it=positiveTrace.erase(it);
        	else
        		++it;
        }
    }
}




cv::softcascade::Trace::Trace(const uint64 ind,const uint octave, const uint level, const Detection& dw, const std::vector<float>& scores, const std::vector<float>& stagesResp, const int classification)
 :index(ind),octaveIndex(octave),levelIndex(level),detection(dw.bb(),dw.confidence,dw.kind), subscores(scores),stages(stagesResp),classType(classification) {localMaxIndex=localMinIndex=0;}


/*
cv::softcascade::DetectorTrace::DetectorTrace(const double mins, const double maxs, const int nsc, const int rej)
: Detector(mins,maxs,nsc,rej) {traceType2Return= LOCALMAXIMUM_TR;}
*/

cv::softcascade::DetectorTrace::DetectorTrace(ParamDetectorFast param)
: Detector(param.minScale,param.maxScale,param.nScales,param.nMS) {traceType2Return= LOCALMAXIMUM_TR;}

cv::softcascade::DetectorTrace::~DetectorTrace() {}

void cv::softcascade::DetectorTrace::detectAtTrace(const int dx, const int dy, const Level& level, const ChannelStorage& storage, std::vector<Trace>& positiveTrace,std::vector<Trace>& negativeTrace, uint levelI)
{

    float detectionScore = 0.f;

//    Level level= *(static_cast<Level*>(lv));
//    ChannelStorage storage= *(static_cast<ChannelStorage*>(stg));


    const SOctave& octave = *(level.octave);

    int stBegin = octave.index * octave.weaks, stEnd = stBegin + octave.weaks;

    std::vector<float> subScores;
    std::vector<float> stages;


    for(int st = stBegin; st < stEnd; ++st)
    {

        const Weak& weak = fields->weaks[st];

        int nId = st * 3;

        // work with root node
        const Node& node = fields->nodes[nId];
        const Feature& feature = fields->features[node.feature];

        cv::Rect scaledRect(feature.rect);

        float threshold = level.rescale(scaledRect, node.threshold, (int)(feature.channel > 6)) * feature.rarea;
        float sum = storage.get(feature.channel, scaledRect);
        int next = (sum >= threshold)? 2 : 1;

        // leaves
        const Node& leaf = fields->nodes[nId + next];
        const Feature& fLeaf = fields->features[leaf.feature];

        scaledRect = fLeaf.rect;
        threshold = level.rescale(scaledRect, leaf.threshold, (int)(fLeaf.channel > 6)) * fLeaf.rarea;
        sum = storage.get(fLeaf.channel, scaledRect);

        int lShift = (next - 1) * 2 + ((sum >= threshold) ? 1 : 0);
        float impact = fields->leaves[(st * 4) + lShift];

        detectionScore += impact;
        stages.push_back(impact);
        subScores.push_back(detectionScore);

        if (detectionScore <= weak.threshold){
        	if(traceType2Return==NEGATIVE_TR ||traceType2Return==NEG_POS_TR){
        		int shrinkage = 4;//(*octave).shrinkage;
        		cv::Rect rect(cvRound(dx * shrinkage), cvRound(dy * shrinkage), level.objSize.width, level.objSize.height);

        		negativeTrace.push_back(cv::softcascade::Trace(
        				static_cast<uint64>(negativeTrace.size()),octave.index,levelI,cv::softcascade::Detection(rect, detectionScore),subScores,stages,Trace::NEGATIVE));
        	}
        	return;
        }
    }

    //content of the function level.addDetection()
    //  level.addDetection(dx, dy, detectionScore, detections);

	// fix me
	int shrinkage = 4;//(*octave).shrinkage;
	cv::Rect rect(cvRound(dx * shrinkage), cvRound(dy * shrinkage), level.objSize.width, level.objSize.height);


	if (detectionScore > 0. ){
		if(traceType2Return!=NEGATIVE_TR)
		positiveTrace.push_back(cv::softcascade::Trace(
				static_cast<uint64>(positiveTrace.size()),octave.index,levelI,cv::softcascade::Detection(rect, detectionScore),subScores,stages,Trace::POSITIVE));

	}
	else if(traceType2Return==NEGATIVE_TR ||traceType2Return==NEG_POS_TR)
		negativeTrace.push_back(cv::softcascade::Trace(
				static_cast<uint64>(negativeTrace.size()),octave.index,levelI,cv::softcascade::Detection(rect, detectionScore),subScores,stages,Trace::NEGATIVE));
}


void cv::softcascade::DetectorTrace::detectNoRoiTrace(const cv::Mat& image, std::vector<Trace>& positiveTrace,std::vector<Trace>& negativeTrace)
{
    Fields& fld = *fields;
    // create integrals
    ChannelStorage storage(image, fld.shrinkage, fld.featureTypeStr);

    typedef std::vector<Level>::const_iterator lIt;
    for (lIt it = fld.levels.begin(); it != fld.levels.end(); ++it)
    {
        const Level& level = *it;

        // we train only 3 scales.
        if (level.origScale > 2.5) break;

        for (int dy = 0; dy < level.workRect.height; ++dy)
        {
            for (int dx = 0; dx < level.workRect.width; ++dx)
            {
                storage.offset = (int)(dy * storage.step + dx);
                detectAtTrace(dx,dy,level,storage,positiveTrace,negativeTrace,it-fld.levels.begin());
            }
        }
    }

    if (traceType2Return != NEGATIVE_TR) DollarNMSTrace(positiveTrace,traceType2Return==LOCALMAXIMUM_TR);
}

uint cv::softcascade::DetectorTrace::nLevels(){return fields->levels.size();}

void cv::softcascade::DetectorTrace::getRejectionThreshols(std::vector< std::vector<float> >& ths){
	ths.clear();

	std::vector<SOctave>::iterator itO;

	std::cout<<"Extraction Rejected Thresholds: "<<std::endl;
	ths=std::vector< std::vector<float> > (fields->octaves.end()-fields->octaves.begin(), std::vector<float>() );

	for (itO=fields->octaves.begin();itO!=fields->octaves.end();++itO){

		int stBegin = itO->index * itO->weaks, stEnd = stBegin + itO->weaks;
		std::cout<<"\tOctave "<<itO-fields->octaves.begin()<<": ";

		ths[itO-fields->octaves.begin()]=std::vector<float>(stEnd-stBegin,0.);
		for(int st = stBegin; st < stEnd; ++st)
		{
			const Weak& weak = fields->weaks[st];
			std::cout<<weak.threshold<<" ";
			ths[itO-fields->octaves.begin()][st-stBegin]=weak.threshold;

		}

		std::cout<<std::endl;
	}
}

void cv::softcascade::DetectorTrace::getInfo4LevelId(int id, cv::Size& workRect, cv::Size& dw, int& octaveId){
	workRect=fields->levels[id].workRect;
	dw=fields->levels[id].objSize;
	octaveId=fields->levels[id].octave->index;

}
void cv::softcascade::DetectorTrace::detectTrace(InputArray _image, InputArray _rois, std::vector<Trace>& positiveTrace,std::vector<Trace>& negativeTrace, int traceType)
{

	//assign the type of trace to return
	traceType2Return=traceType;

    // only color images are suppered
    cv::Mat image = _image.getMat();
    CV_Assert(image.type() == CV_8UC3);

    Fields& fld = *fields;
    fld.calcLevels(image.size(),(float) minScale, (float)maxScale, scales);

    // initialize result's vector
    positiveTrace.clear();
    negativeTrace.clear();

    if (_rois.empty())
        return detectNoRoiTrace(image, positiveTrace,negativeTrace);

    int shr = fld.shrinkage;

    cv::Mat roi = _rois.getMat();
    cv::Mat mask(image.rows / shr, image.cols / shr, CV_8UC1);

    mask.setTo(cv::Scalar::all(0));
    cv::Rect* r = roi.ptr<cv::Rect>(0);
    for (int i = 0; i < (int)roi.cols; ++i)
        cv::Mat(mask, cv::Rect(r[i].x / shr, r[i].y / shr, r[i].width / shr , r[i].height / shr)).setTo(cv::Scalar::all(1));

    // create integrals
    ChannelStorage storage(image, shr, fld.featureTypeStr);

    typedef std::vector<Level>::const_iterator lIt;
    for (lIt it = fld.levels.begin(); it != fld.levels.end(); ++it)
    {
         const Level& level = *it;

        // we train only 3 scales.
        if (level.origScale > 2.5) break;

         for (int dy = 0; dy < level.workRect.height; ++dy)
         {
             uchar* m  = mask.ptr<uchar>(dy);
             for (int dx = 0; dx < level.workRect.width; ++dx)
             {
                 if (m[dx])
                 {
                     storage.offset = (int)(dy * storage.step + dx);
                     detectAtTrace(dx, dy, level, storage, positiveTrace,negativeTrace,it-fld.levels.begin());
                 }
             }
         }
    }

    if (traceType2Return != NEGATIVE_TR) DollarNMSTrace(positiveTrace,traceType2Return==LOCALMAXIMUM_TR);
}





