createTuningJobFiles.py all \
              -oJConf SAE_config \
              --nInits 1 \
              --neuronBounds 10 10 \
              -outCross crossValid_SAE_jackknife \
              --method "JackKnife" \
              -outPP ppFile_preproc \
              -pp_nEt 4 \
              -pp_nEta 9 \
              -ppCol "[Norm1(),StackedAutoEncoder({'hidden_neurons' : [$1], 'aetype' : 'vanilla'}),StackedAutoEncoder({'hidden_neurons' : [$2], 'aetype' : 'vanilla'}),StackedAutoEncoder({'hidden_neurons' : [$3], 'aetype' : 'vanilla'}),StackedAutoEncoder({'hidden_neurons' : [$4], 'aetype' : 'vanilla'}),StackedAutoEncoder({'hidden_neurons' : [$5], 'aetype' : 'vanilla'}),SAE_FineTuning()]"
              #PCA({'energy' : 66})]"
              #,NLPCA(level = self.level, nlpcs=47, nmapping=71)] ))
              #,PCA(level = self.level, energy = 70)] )) #,
              #StackedAutoEncoder({'hidden_neurons' : [60], 'aetype' : 'contractive'})] )) #,
              #StackedAutoEncoder(level=self.level,hidden_neurons=[30],aetype='vanilla')] )) #, SAE_FineTuning()] ))
              #StackedAutoEncoder(level = self.level,hidden_neurons=[50], aetype='vanilla'),
              #StackedAutoEncoder(level = self.level,hidden_neurons=[40], aetype='vanilla'),
              #StackedAutoEncoder(level = self.level,hidden_neurons=[30], aetype='vanilla'),
              #StackedAutoEncoder(level = self.level,hidden_neurons=[20], aetype='vanilla'),
