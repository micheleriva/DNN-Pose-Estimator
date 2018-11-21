package main

import (
	"encoding/json"
	"fmt"
	"gocv.io/x/gocv"
	"image"
)

type output struct {
	Desc string `json:"desc"`
	Points []image.Point `json:"points"`
}

var net *gocv.Net
var images chan *gocv.Mat
var poses chan [][]image.Point
var pose [][]image.Point

var deviceID = 0
var proto = "./src/models/pose_iter_440000.caffemodel"
var model = "./src/models/openpose_pose_coco.prototxt"
var backend = gocv.NetBackendDefault
var target = gocv.NetTargetCPU

func main() {

	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Printf("Error opening device %v\n", deviceID)
		return
	}
	defer webcam.Close()

	img := gocv.NewMat()
	defer img.Close()

	n := gocv.ReadNet(model, proto)
	net = &n
	if net.Empty() {
		fmt.Printf("Error reading network model from : %v %v\n", model, proto)
		return
	}
	defer net.Close()

	net.SetPreferableBackend(gocv.NetBackendType(backend))
	net.SetPreferableTarget(gocv.NetTargetType(target))

	images = make(chan *gocv.Mat, 1)
	poses = make(chan [][]image.Point)

	if ok := webcam.Read(&img); !ok {
		fmt.Printf("Error cannot read device %v\n", deviceID)
		return
	}

	processFrame(&img)

	go performDetection()

	for {

		if ok := webcam.Read(&img); !ok {
			fmt.Printf("Device closed: %v\n", deviceID)
			return
		}

		if img.Empty() {
			continue
		}

		select {
			case pose = <-poses:
				processFrame(&img)
			default:
		}

		printOutput(&img)

	}
}

func processFrame(i *gocv.Mat) {
	frame := gocv.NewMat()
	i.CopyTo(&frame)
	images <- &frame
}

func performDetection() {

	for {

		frame := <-images
		blob := gocv.BlobFromImage(*frame, 1.0/255.0, image.Pt(368, 368), gocv.NewScalar(0, 0, 0, 0), false, false)
		net.SetInput(blob, "")
		prob := net.Forward("")

		var midx int

		s := prob.Size()

		nparts, h, w := s[1], s[2], s[3]

			switch nparts {
				case 19:
					midx = 0
					nparts = 18
				case 16:
					midx = 1
					nparts = 15
				case 22:
					midx = 2
				default:
					fmt.Println("there should be 19 parts for the COCO model, 16 for MPI, or 22 for the hand model")
					return
			}

			pts := make([]image.Point, 22)

			for i := 0; i < nparts; i++ {
				pts[i] = image.Pt(-1, -1)
				heatmap, _ := prob.FromPtr(h, w, gocv.MatTypeCV32F, 0, i)

				_, maxVal, _, maxLoc := gocv.MinMaxLoc(heatmap)
				if maxVal > 0.1 {
					pts[i] = maxLoc
				}
				heatmap.Close()
			}

			sX := int(float32(frame.Cols()) / float32(w))
			sY := int(float32(frame.Rows()) / float32(h))

			results := [][]image.Point{}
			for _, p := range PosePairs[midx] {
				a := pts[p[0]]
				b := pts[p[1]]

				if a.X <= 0 || a.Y <= 0 || b.X <= 0 || b.Y <= 0 {
					continue
				}

				a.X *= sX
				a.Y *= sY
				b.X *= sX
				b.Y *= sY

				results = append(results, []image.Point{a, b})
			}
			prob.Close()
			blob.Close()
			frame.Close()

			poses <- results
		}

}


func printOutput(frame *gocv.Mat) {

	for _, pts := range pose {

		res := &output{
			Desc: "Output",
			Points: []image.Point{ pts[0], pts[1] }}

		response, _ := json.Marshal(res)

		fmt.Println(response)
	}

}

var PosePairs = [3][20][2]int{
	{
		{1, 2}, {1, 5}, {2, 3},
		{3, 4}, {5, 6}, {6, 7},
		{1, 8}, {8, 9}, {9, 10},
		{1, 11}, {11, 12}, {12, 13},
		{1, 0}, {0, 14},
		{14, 16}, {0, 15}, {15, 17},
	},
	{
		{0, 1}, {1, 2}, {2, 3},
		{3, 4}, {1, 5}, {5, 6},
		{6, 7}, {1, 14}, {14, 8}, {8, 9},
		{9, 10}, {14, 11}, {11, 12}, {12, 13},
	},
	{
		{0, 1}, {1, 2}, {2, 3}, {3, 4},
		{0, 5}, {5, 6}, {6, 7}, {7, 8},
		{0, 9}, {9, 10}, {10, 11}, {11, 12},
		{0, 13}, {13, 14}, {14, 15}, {15, 16},
		{0, 17}, {17, 18}, {18, 19}, {19, 20},
	}}