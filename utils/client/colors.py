colorSchemes = {
    'Set1': ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
}

class ColorScale:
    def __init__(self, colorScheme='Set1'):
        if colorScheme not in colorSchemes:
            colorScheme = 'Set1'
        
        self.colors = colorSchemes[colorScheme]
        self.colorMap = {}
    
    def get_color(self, value):
        if value in self.colorMap:
            return self.colorMap[value]

        next_index = len(self.colorMap) % len(self.colors)
        color = self.colors[next_index]
        self.colorMap[value] = color
        return color
    
    def get_values(self):
        return self.colorMap.keys()

    def __len__(self):
        return len(self.colors)