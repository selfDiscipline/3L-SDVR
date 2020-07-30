
 
  /*var vectorLayer1 = new ol.layer.Vector({
    source: new ol.format.GeoJSON({
      url:routeUrl,
      crossDomain: true,
      }),
    style: new ol.style.Style({
      stroke: new ol.style.Stroke({
        color: 'blue',
        width: 4
        })
      }),
    title: "Route",
    name: "Route"
    });*/
    
  /*var vectorLayer = new ol.layer.Vector({
    source: new ol.source.GeoJSON({url:geojs_url}),
    style: new ol.style.Style({stroke: new ol.style.Stroke({color: 'red',width: 4})}),
    title: "Route",
    name: "Route"
    });
    */
    
  
  var map = new ol.Map({
    layers: [
    new ol.layer.Tile({source: new ol.source.OSM()}),
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
  center: ol.proj.fromLonLat([51.387505, 35.701854]),
  zoom: 12})});

  map.on('singleclick', function (evt) {
    console.log(evt.coordinate);
    console.log(ol.proj.transform(evt.coordinate, 'EPSG:3857', 'EPSG:4326'));
  });


  function addRoute() {
    var userHostName;
    if (window.location.hostname == "192.168.100.96") {
      userHostName = "http://192.168.100.96";
    }
    else {
      userHostName = window.location.protocol + "//" + window.location.hostname;
    }
    console.log(window.location.protocol)
    var final_route
    $.ajax({
      async:false,
      url: userHostName + ':8082/api/routes',
      success: function(data) {
        final_route = data;
      }
    });


    // var baseUrl = 'http://localhost:8082/api/directions/';

    // var routeUrl =baseUrl + '35.69974,51.34284&35.75617,51.51283/';
  
    // var polyline_url = [routeUrl].join('');
    // var polyline;
    // $.ajax({
    //     async:false,
    //     url: polyline_url,
    //     success: function(data) {
    //         polyline = data;
    //     }
    // });

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
                  image: new ol.style.Icon({
                      anchor: [0.5, 1],
                      src: 'data/icon.png'
                  })
              }),
              'geoMarker': new ol.style.Style({
                  image: new ol.style.Circle({
                      radius: 7,
                      fill: new ol.style.Fill({color: 'black'}),
                      stroke: new ol.style.Stroke({
                      color: 'white', width: 2
                      })
                  })
              })
          };

    var vectorLayer = new ol.layer.Vector({
        source: new ol.source.Vector({
            features: [routeFeature]
        }),
        style: function(feature) {
            return styles[feature.get('type')];
        }
    });


    var geoMarker = new ol.Feature({
        type: 'geoMarker',
        geometry: new ol.geom.Point(routeCoords[0])
    });

    map.getLayers().push(vectorLayer);
    }