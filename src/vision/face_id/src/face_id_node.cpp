//OpenCV libraries
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//ROS
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <pi_face_tracker/Face.h>
#include <pi_face_tracker/Faces.h>
#include <face_id/face_id.h>
#include <face_id/faces_ids.h>

#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <sensor_msgs/image_encodings.h>
//openbr
#include <openbr/openbr_plugin.h>
#include <QMutex>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <mutex>
using namespace std;
using namespace cv;

//ASSUMES SAME IMAGE RESOLUTION IMAGE REFERENCE AS CMT TRACKER

//load gallery txt file with file,person_name strings to create a mapping
//subscribe to camera and tracked faces
//if recognized faces in camera image overlap with tracked faces then map them in ros msg

QSharedPointer<br::Transform> transform = br::Transform::fromAlgorithm("FaceRecognition");
QSharedPointer<br::Distance> distance = br::Distance::fromAlgorithm("FaceRecognition");
std::mutex z_mutex;

string face_cascade_name = "haarcascade_frontalface_alt.xml";
string gallery_path;
std::map<string,string> file2name;
//String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
//CascadeClassifier eyes_cascade;
std::vector<Rect> faces;
struct face_pix{
  int id;
  int fx;
  int fy;
};
std::vector<face_pix>frv;

std::vector<cv::Mat> getFaces(cv::Mat frame_gray)
{
  //
  std::vector<cv::Mat> fcs;
  equalizeHist( frame_gray, frame_gray );
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
  for( size_t i = 0; i < faces.size(); i++ )
  {
    //
    Mat faceROI = frame_gray( faces[i] );
    fcs.push_back(faceROI);
  }
  return fcs;
}

bool is_point_in_rect(float x,float y,Rect rct)
{
  return ((x>=rct.x) && (x<=rct.x+rct.width) && (y>=rct.y) && (y<=rct.y+rct.height));
}
bool rects_overlap(Rect r1,Rect r2)
{
  return (
    is_point_in_rect(r1.x,r1.y,r2)||
    is_point_in_rect(r1.x+r1.width,r1.y,r2)||
    is_point_in_rect(r1.x,r1.y+r1.height,r2)||
    is_point_in_rect(r1.x+r1.width,r1.y+r1.height,r2)
  )
}

int get_overlap_id(Rect rct)
{
  std::lock_guard<std::mutex> guard(z_mutex);
  for (int  i=0;i<frv.size();i++)
  {
    if (is_point_in_rect(frv[i].fx,frv[i].fy,rct))
    {
      return frv[i].id;
    }
  }
  return 0;
}

std::string get_name_from_gallery(int idx)
{
  //check if file.name is same as name in gallery
  std::map<string,string>::iterator it;
  it=file2name.find(target[idx].file.name);
  if (it != file2name.end())
    return file2name[target[idx].file.name];
  return "stranger";
}

void image_cb(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::RGB8);
  cv::Mat img = cv_ptr->image;
  cv::Mat img_gray;
  //ROS_INFO("I heard: [%s]", msg->data.c_str());
  //find faces and pass each face with co-ordinates for recognition
  cv::cvtColor(img, img_gray, CV_BGR2GRAY);
  std::vector<cv::Mat> fcs=getFaces(img_gray);
  face_id_msgs::faces_ids fc_ids;
  for( size_t i = 0; i < fcs.size(); i++ )
  {
    br::Template query(fcs[i]);
    query >> *transform;
    // Compare templates
    QList<float> scores = distance->compare(target, query);
    //FORM ROS MSG
    float score=scores[0];
    int index=0;
    for (int a=1;a<scores.size();a++)
    {
      if (scores[a]>score)
      {
        score=scores[a];
        index=a;
      }
    }
    //update message to send
    face_id_msgs::face_id fid;
    fid.face_id=get_overlap_id(faces[index]);
    fid.name=(score>0.1)?get_name_from_gallery(index):"stranger";
    fid.confidence=score;
    fc_ids.push_back(fid);
  }
  //publish message
}

string trim(const string& str)
{
    size_t first = str.find_first_not_of(' ');
    if (string::npos == first)
    {
        return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

void map_image2name(string line)
{
  //split line in 2
  string file,name;
  unsigned int pos = line.find( "," );
  if (pos<1) return;
  file= trim(line.substr( 0, pos ));
  name= trim(line.substr(pos+1,line.length()-pos-1));
  file2name[file]=name;
}

void faces_cb(const pi_face_tracker_msgs::Faces msg)
{
  const float pic_width=640.0;//pixels
  const float pic_height=480.0;
  const float fov=1.42;//radians
  const float fsz=0.17;//meters face width
  const double k_const = (double)pic_width / (double) (2.0*(tan(fov/2.0)));
  float xx,yy,zz;
  face_pix fr;
  //need tracking id and rect
  std::lock_guard<std::mutex> guard(z_mutex);
  frv.clear();
  for (int i=0;i<msg.faces.size();i++)
  {
    //
    fr.id=msg.faces[i].id;
    //convert x,y to pixels from 3d xyz
    xx=msg.faces[i].point.x;//y
    yy=msg.faces[i].point.y;//z
    zz=msg.faces[i].point.z;//x
    double dp=xx/k_const;
    fr.fx=(pic_width/2.0)-(zz/dp);
    fr.fy=(pic_height/2.0)-(yy/dp);
    frv.push_back(fr);
  }
}

void load_name_image_map()
{
  //open gallery file from path
  string line;
  ifstream gallery_file(gallery_path);
  gallery_file.open();
  while ( getline (myfile,line) )
  {
    map_image2name(line);
  }
  gallery_file.close();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "faceid");
  ros::NodeHandle n;
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
  //if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

  //Load image name map
  load_name_image_map();
  //
  br::Context::initialize(argc, argv);

  // Retrieve classes for enrolling and comparing templates using the FaceRecognition algorithm
  transform = br::Transform::fromAlgorithm("FaceRecognition");
  distance = br::Distance::fromAlgorithm("FaceRecognition");

  // Initialize templates
  br::TemplateList target = br::TemplateList::fromGallery("../data/MEDS/img");

  // Enroll templates
  br::Globals->enrollAll = true; // Enroll 0 or more faces per image
  target >> *transform;
  br::Globals->enrollAll = false;
  //
  ros::Subscriber sub = n.subscribe("/camera/image_raw", 2, image_cb);
  ros::Subscriber sub_face = n.subscribe("/camera/face_locations", 1, &faces_cb, this);
  //need rect of face
  ros::spin();
  br::Context::finalize();
  return 0;
}
