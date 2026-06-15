import osmium

# Test with_areas for landuse
fp = osmium.FileProcessor(r'dataset\infra\us\delaware-260612.osm.pbf').with_areas().with_locations()
count = 0
for obj in fp:
    if hasattr(obj, 'outer_rings'):
        lu = obj.tags.get('landuse', '')
        if lu in ['residential', 'farmland', 'forest']:
            rings = list(obj.outer_rings())
            print(f'Area: landuse={lu}, outer_rings={len(rings)}')
            if rings:
                pts = [(n.location.lat, n.location.lon) for n in rings[0] if n.location.valid()]
                print(f'  Points: {len(pts)}, first: {pts[0] if pts else None}')
            count += 1
            if count >= 3:
                break
print('Areas test done, found:', count)
