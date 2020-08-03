

var center = ol.proj.fromLonLat([51.417505, 35.751854])
var map = new ol.Map({
  layers: [
    new ol.layer.Tile({ source: new ol.source.XYZ({
      url: 'http://mt0.google.com/vt/lyrs=m@221097413,traffic&hl=en&x={x}&y={y}&z={z}' })
    })
    //vectorLayer
  ],
  target: 'map',
  controls: ol.control.defaults({
    attributionOptions: /** @type
  {olx.control.AttributionOptions} */ ({
        collapsible: false
      })
  }),
  view: new ol.View({
    center: center,
    zoom: 12
  })
});

map.on('singleclick', function (evt) {
  console.log(evt.coordinate);
  console.log(ol.proj.transform(evt.coordinate, 'EPSG:3857', 'EPSG:4326'));
});


function addRoute(collection_points, vehicle_number, cargoe_number) {

  //----------- Send request to API and run the 3L_SDVRP algorithm ------------//
  document.getElementById("myTextArea").value = "Please wait";
  var userHostName = window.location.protocol + "//" + window.location.hostname;
  var final_route
  $.ajax({
    async: false,
    url: userHostName + ':8082/api/routes/' + collection_points + "&" + vehicle_number + "&" + cargoe_number,
    success: function (data) {
      final_route = data;
    }
  });
  console.log(final_route)

  //------------ Clear previous layers on the map----------//
  map.setLayerGroup(new ol.layer.Group());
  map.addLayer(new ol.layer.Tile({ source: new ol.source.XYZ({
    url: 'http://mt0.google.com/vt/lyrs=m@221097413,traffic&hl=en&x={x}&y={y}&z={z}' }) }))

  // --------- show the collection points ------//

  addCollectionPoints(final_route[1], collection_points)

  // ------------- get the routes plans and create layer ----------------//

  addRouteplans(final_route[0][0])

  var next_rnd_btn = document.getElementById("next_round")
  var prev_rnd_btn = document.getElementById("previous_round")
  
  var routplanneitarator = 0

  function checkForVisibility(){
    if (routplanneitarator+1 == final_route[0].length){
      next_rnd_btn.disabled = true
    }
    else{
      next_rnd_btn.disabled = false
    }
    if (routplanneitarator == 0){
      prev_rnd_btn.disabled = true
    }
    else{
      prev_rnd_btn.disabled = false
    }
  }
  checkForVisibility()

  next_rnd_btn.addEventListener('click', function() { addNextRoute() });
  prev_rnd_btn.addEventListener('click', function() { addPrevRoute() });

  function addNextRoute(){
    console.log("hi",routplanneitarator)
    map.setLayerGroup(new ol.layer.Group());
    map.addLayer(new ol.layer.Tile({ source: new ol.source.XYZ({
      url: 'http://mt0.google.com/vt/lyrs=m@221097413,traffic&hl=en&x={x}&y={y}&z={z}' })
    }))
    routplanneitarator = routplanneitarator + 1
    addCollectionPoints(final_route[1], collection_points)
    addRouteplans(final_route[0][routplanneitarator])
    checkForVisibility()
    var round_num_nxt = routplanneitarator + 2
    var round_num_prev = routplanneitarator + 1
    next_rnd_btn.textContent = 'Round ' + round_num_nxt
    prev_rnd_btn.textContent = 'Round ' + round_num_prev
  }

  function addPrevRoute(){
    console.log("by",routplanneitarator)
    map.setLayerGroup(new ol.layer.Group());
    map.addLayer(new ol.layer.Tile({ source: new ol.source.XYZ({
      url: 'http://mt0.google.com/vt/lyrs=m@221097413,traffic&hl=en&x={x}&y={y}&z={z}' })
    }))
    routplanneitarator = routplanneitarator - 1
    addCollectionPoints(final_route[1], collection_points)
    addRouteplans(final_route[0][routplanneitarator])
    checkForVisibility()
    var round_num_nxt = routplanneitarator + 2
    var round_num_prev = routplanneitarator + 1
    next_rnd_btn.textContent = 'Round ' + round_num_nxt
    prev_rnd_btn.textContent = 'Round ' + round_num_prev
  }
  

  // show details on the text box
  document.getElementById("myTextArea").value = "Total rounds : " + final_route[0].length;
  for (var i = 0; i < final_route[2].length; i++) {
    document.getElementById("myTextArea").value = document.getElementById("myTextArea").value + "\n" 
    + "Round(" + i +"): " + "\n" + JSON.stringify(final_route[2][i])
  }
  //------------ sow details on Modal -----------//
  
  // Get the modal
  
  var detail_Modal = document.getElementById("detail_Modal");
  var detail_modal_Btn = document.getElementById("detail_modal_Btn");
  var modal_close = document.getElementById("modal_close");
  var detai_modal_content = document.getElementById("detail_modal_content_2");
  
  // Emty the modal if there is eny detail from the previous run
  detai_modal_content.innerHTML = '';


  var detais_paragraph_0 = document.createElement("p");
  var p_content_0 = document.createTextNode("* The process is done within ( " + final_route[2].length + " ) rounds. The details about vehicle's capacity and cargoes which are allocated to each vehicle in each round are shown in the following.");
  var p_content_1 = document.createTextNode("** First element of each cargoe is the number of it's corresponding collection point and the second element is it's number. e.g. ( 5 , 1 ) mean the first cargoe of collection point 5 . ");
  detais_paragraph_0.appendChild(p_content_0);
  var br = document.createElement("br");
  detais_paragraph_0.appendChild(br)
  detais_paragraph_0.appendChild(p_content_1);
  detai_modal_content.appendChild(detais_paragraph_0)
  
  // Go through the json object of the details
  for (var i = 0; i < final_route[2].length; i++) {
    var detais_paragraph = document.createElement("p");
    var p_content_2 = document.createTextNode("Round ( " + i + " ) : ");
    detais_paragraph.appendChild(p_content_2);
    for (var property in final_route[2][i]) {
      if (final_route[2][i].hasOwnProperty(property)) {
        var br = document.createElement("br");
        detais_paragraph.appendChild(br)
        var p_content_3 = document.createTextNode(property + " : ");
        detais_paragraph.appendChild(p_content_3);
        for (var property_2 in final_route[2][i][property]) {
          if (final_route[2][i][property].hasOwnProperty(property_2)) {
            var br = document.createElement("br");
            detais_paragraph.appendChild(br)
            var p_content_4 = document.createTextNode(property_2 + " : ");
            detais_paragraph.appendChild(p_content_4);
            for (var property_3 in final_route[2][i][property][property_2]) {
              if (final_route[2][i][property][property_2].hasOwnProperty(property_3)) {
                var br = document.createElement("br");
                detais_paragraph.appendChild(br)
                var p_content_5 = document.createTextNode(final_route[2][i][property][property_2][property_3]);
                detais_paragraph.appendChild(p_content_5);
              }
            }
          }
        }
      }
    }
    detai_modal_content.appendChild(detais_paragraph);
  }

  // When the user clicks the button, open the modal 
  detail_modal_Btn.onclick = function() {
    detail_Modal.style.display = "block";
  }
  // When the user clicks on <span> (x), close the modal
  modal_close.onclick = function() {
    detail_Modal.style.display = "none";
  }
  // When the user clicks anywhere outside of the modal, close it
  window.onclick = function(event) {
    if (event.target == detail_Modal) {
      detail_Modal.style.display = "none";
    }
  }

}

function addCollectionPoints(final_route, collection_points){
  var source_collection_points = new ol.source.Vector({
    wrapX: false,
  });
  var collection_points_loc = final_route.slice(0, parseInt(collection_points) + 1)
  addRandomFeature(collection_points_loc, source_collection_points)
  var vector_collection_points = new ol.layer.Vector({
    source: source_collection_points,
    style: new ol.style.Style({
      image: new ol.style.Circle({
        radius: 8,
        fill: new ol.style.Fill({ color: 'red' }),
      }),
    }),
  });
  map.addLayer(vector_collection_points);
}

function addRouteplans(final_route){

  var route = /** @type {import("../src/ol/geom/LineString.js").default} */ (new ol.format.Polyline(
    {
      factor: 1e5,
    }
  ).readGeometry(final_route, {
    dataProjection: 'EPSG:4326',
    featureProjection: 'EPSG:3857',
  }));

  var routeCoords = route.getCoordinates();
  var routeLength = routeCoords.length;
  var routeFeature = new ol.Feature({
    type: 'route',
    geometry: route
  });
  var styles = {
    'route': new ol.style.Style({
      stroke: new ol.style.Stroke({
        width: 6, color: [24, 170, 13, 0.8]
      })
    }),
    'icon': new ol.style.Style({
      image: new ol.style.Circle({
        radius: 8,
        fill: new ol.style.Fill({ color: 'blue' }),
      }),
    }),
    // new ol.style.Style({
    //     image: new ol.style.Icon({
    //         anchor: [0.5, 1],
    //         src: '../static/img/c-p.png',
    //     })
    // }),
    'geoMarker': new ol.style.Style({
      image: new ol.style.Circle({
        radius: 7,
        fill: new ol.style.Fill({ color: 'black' }),
        stroke: new ol.style.Stroke({
          color: 'white', width: 2
        })
      })
    })
  };

  var geoMarker = new ol.Feature({
    type: 'geoMarker',
    geometry: new ol.geom.Point(routeCoords[0])
  });
  var startMarker = new ol.Feature({
    type: 'icon',
    geometry: new ol.geom.Point(routeCoords[0]),
  });
  var endMarker = new ol.Feature({
    type: 'icon',
    geometry: new ol.geom.Point(routeCoords[routeLength - 2]),
  });

  var vectorLayer = new ol.layer.Vector({
    source: new ol.source.Vector({
      features: [routeFeature, geoMarker, startMarker]
    }),
    style: function (feature) {
      // hide geoMarker if animation is active
      if (animating && feature.get('type') === 'geoMarker') {
        return null;
      }
      return styles[feature.get('type')];
    }
  });
  
  map.addLayer(vectorLayer);

  
  var animating = false;
  var speed, now;
  var speedInput = document.getElementById('speed');
  var startButton = document.getElementById('start-animation');

  var moveFeature = function (event) {
    var vectorContext = ol.render.getVectorContext(event);
    var frameState = event.frameState;

    if (animating) {
      var elapsedTime = frameState.time - now;
      // here the trick to increase speed is to jump some indexes
      // on lineString coordinates
      var index = Math.round((speed * elapsedTime) / 1000);

      if (index >= routeLength) {
        stopAnimation(true);
        return;
      }

      var currentPoint = new ol.geom.Point(routeCoords[index]);
      var feature = new ol.Feature(currentPoint);
      vectorContext.drawFeature(feature, styles.geoMarker);
    }
    // tell OpenLayers to continue the postrender animation
    map.render();
  };

  function startAnimation() {
    if (animating) {
      stopAnimation(false);
    } else {
      animating = true;
      now = new Date().getTime();
      speed = speedInput.value;
      startButton.textContent = 'Cancel Animation';
      // hide geoMarker
      geoMarker.setStyle(null);
      // just in case you pan somewhere else
      // map.getView().setCenter(center);
      vectorLayer.on('postrender', moveFeature);
      map.render();
    }
  }

  /**
   * @param {boolean} ended end of animation.
   */
  function stopAnimation(ended) {
    animating = false;
    startButton.textContent = 'Start Animation';

    // if animation cancelled set the marker at the beginning
    var coord = ended ? routeCoords[routeLength - 1] : routeCoords[0];
    var geometry = geoMarker.getGeometry();
    geometry.setCoordinates(coord);
    //remove listener
    vectorLayer.un('postrender', moveFeature);
  }

  startButton.addEventListener('click', startAnimation, false);
}


function addRandomFeature(final_route, source_collection_points) {
  var arrayLength = final_route.length;
  for (var i = 0; i < arrayLength; i++) {
    var geom = new ol.geom.Point(ol.proj.fromLonLat([final_route[i][1], final_route[i][0]]));
    var feature = new ol.Feature(geom);
    source_collection_points.addFeature(feature);
  }
}