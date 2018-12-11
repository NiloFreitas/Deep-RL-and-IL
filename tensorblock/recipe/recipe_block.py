
class recipe_block:

###### Repeat Block
    def repBlock( self , **args ):

        block_pars = { **self.defs_block , **args }
        src , dst , type = block_pars['src'] , block_pars['dst'] , block_pars['type']

        list_src = src.split( '/' )
        src = self.iterate( list_src )
        src = src[ list_src[-1] ]

        list_dst = dst.split( '/' )
        dst = self.iterate( list_dst )
        dst.addBlock( list_dst[-1] )
        dst = dst[ list_dst[-1] ]

        for label , name in src.order:

            if label == 'block':
                self.repeat( src = '/' + src.folder + name ,
                             dst = '/' + dst.folder + name , type = type )

            if label == 'input':

                pars = src.pars( name ).copy()
                if type is 'share' and block_pars['mod_inputs']:
                    pars[type] = '/' + src.folder + pars['name']
                dst.addInput( **pars )

            if label == 'variable':

                pars = src.pars( name ).copy()
                if type is not None and block_pars['mod_variables']:
                    pars[type] = '/' + src.folder + pars['name']
                dst.addVariable( **pars )

            if label == 'layer':

                pars = src.pars( name ).copy()
                if type is not None and block_pars['mod_layers']:
                    pars[type] = '/' + src.folder + pars['name']
                dst.addLayer( **pars )

            if label == 'operation' and not block_pars['no_ops']:

                pars = src.pars( name ).copy()
                dst.addOperation( **pars )

        return dst

###### Copy Block
    def copyBlock( self , **args ):
        pars = { **self.defs_block , **args } ; pars['type'] = 'copy'
        return self.repBlock( **pars )

###### Share Block
    def shareBlock( self , **args ):
        pars = { **self.defs_block , **args } ; pars['type'] = 'share'
        return self.repBlock( **pars )
