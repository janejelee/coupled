/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2005 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2005, 2006
 */



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/iterative_inverse.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/lac/sparse_ilu.h>

#include <fstream>
#include <iostream>

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/base/tensor_function.h>

namespace Step20
{
  using namespace dealii;
    using namespace numbers;
  namespace data
  {
  	  const int problem_degree = 1;
  	  const int refinement_level = 4;
  	  const int dimension = 2;
  			  
      const double rho_f = 2.5;
      const double eta = 1.0;
      
      const double top = 1.0;
      const double bottom = 0.0;      
      const double left = 0.0;
      const double right = PI;
     
      const double lambda = 0.7;
      const double perm_const = 0.3;
      
      const double k = 0.3;
      const double phi = 0.7;


  }

  template <int dim>
  class MixedLaplaceProblem
  {
  public:
    MixedLaplaceProblem (const unsigned int degree);
    void run ();

  private:
    void make_grid_and_dofs ();
    void assemble_pf_system ();
    void assemble_vf_system ();
    void solve ();
    void compute_errors () const;
    void output_results () const;

    Triangulation<dim>   triangulation;
    
    
    const unsigned int   pf_degree;
    FESystem<dim>        pf_fe;
    DoFHandler<dim>      pf_dof_handler;
    ConstraintMatrix     pf_constraints;
    
    BlockSparsityPattern      pf_sparsity_pattern;
    BlockSparseMatrix<double> pf_system_matrix;
    BlockVector<double>       pf_solution;
    BlockVector<double>       pf_system_rhs;
    


    const unsigned int 	vf_degree;
    FESystem<dim>		vf_fe;
    DoFHandler<dim>     vf_dof_handler;
    ConstraintMatrix    vf_constraints;
    
    SparsityPattern      vf_sparsity_pattern;
    SparseMatrix<double> vf_system_matrix;
    SparseMatrix<double> vf_mass_matrix;

    Vector<double>       vf_solution;
    Vector<double>       vf_system_rhs;
    
    
   
  
  };

    
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };


  template <int dim>
  class PressureBoundaryValues : public Function<dim>
  {
  public:
    PressureBoundaryValues () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };



  template <int dim>
  double RightHandSide<dim>::value (const Point<dim>  &p,
                                    const unsigned int /*component*/) const
  {

      const double permeability = data::perm_const;
    		  //(0.5*std::sin(3*p[0])+1)* std::exp( -100.0*(p[1]-0.5)*(p[1]-0.5) );
      
      
      return 2.0* data::lambda * permeability * data::rho_f*p[1];
  }



  template <int dim>
  double PressureBoundaryValues<dim>::value (const Point<dim>  &p,
                                             const unsigned int /*component*/) const
  {
	  // This is the dirichlet condition for the top
	  
   return -2./3*data::rho_f;
  }

  template <int dim>
  void negunitz (const std::vector<Point<dim> > &points,
                        std::vector<Tensor<1, dim> >   &values)
  {

      for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
      {

              values[point_n][0] = 0.0;
              values[point_n][1] = -1.0;
      }
      
  }

  	  ///////////////////////////////////////////////
  template <int dim>
  void rock_vel (const std::vector<Point<dim> > &points,
                std::vector<Tensor<1, dim> >   &values)
  {
      
      for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
      {
          
          values[point_n][0] = points[point_n][0]*points[point_n][0] +
                                      points[point_n][1]*points[point_n][1];
          values[point_n][1] = -points[point_n][1]*points[point_n][0];
      }
      
  }


  template <int dim>
  class ExactSolution_pf : public Function<dim>
  {
  public:
    ExactSolution_pf () : Function<dim>(dim+1) {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };

  template <int dim>
  void
  ExactSolution_pf<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {
    Assert (values.size() == dim+1,
            ExcDimensionMismatch (values.size(), dim+1));

      
      const double permeability = data::perm_const;
      
    values(0) = 0.0;
      values(1) = data::lambda*data::rho_f*(1-p[1]*p[1])*permeability;
    values(2) = -data::rho_f*(p[1] - (1.0/3.0)*p[1]*p[1]*p[1]);
  }


  template <int dim>
  class ExactSolution_vf : public Function<dim>
  {
  public:
    ExactSolution_vf () : Function<dim>(dim+1) {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };



  template <int dim>
  void
  ExactSolution_vf<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {


      values(0) = p[0]*p[0] + p[1]*p[1] - data::lambda*data::k*(1.0/data::phi) * (std::sin(p[0])+p[1]); 
      values(1) = -p[0]*p[1] - data::lambda*data::k*(1.0/data::phi) * (std::cos(p[1]) + p[0] + data::rho_f);
  }


    

  template <int dim>
  class KInverse : public TensorFunction<2,dim>
  {
  public:
    KInverse () : TensorFunction<2,dim>() {}

    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const;
  };


    template <int dim>
    void
    KInverse<dim>::value_list (const std::vector<Point<dim> > &points,
                               std::vector<Tensor<2,dim> >    &values) const
    {
        
         Assert (points.size() == values.size(),
          
         
          ExcDimensionMismatch (points.size(), values.size()));
        for (unsigned int p=0; p<points.size(); ++p)
        {
            values[p].clear ();

            const double permeability = data::perm_const;
                
            for (unsigned int d=0; d<dim; ++d)
                values[p][d][d] = 1./permeability;
        }
    }
    
    

  template <int dim>
  class K : public TensorFunction<2,dim>
  {
  public:
    K () : TensorFunction<2,dim>() {}

    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const;
  };


  template <int dim>
  void
  K<dim>::value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const
  {
    Assert (points.size() == values.size(),
            ExcDimensionMismatch (points.size(), values.size()));

    for (unsigned int p=0; p<points.size(); ++p)
      {
          values[p].clear ();
          
          
          Assert (points.size() == values.size(),
                  ExcDimensionMismatch (points.size(), values.size()));
          for (unsigned int p=0; p<points.size(); ++p)
          {
              values[p].clear ();/*
                                  const double distance_to_flowline
                                   = std::fabs(points[p][1]-0.2*std::sin(10*points[p][0]));*/
              
             
              const double permeability = data::perm_const;
              
              for (unsigned int d=0; d<dim; ++d)
                  values[p][d][d] = permeability;
          }

      }
  }

    
  template <int dim>
  MixedLaplaceProblem<dim>::MixedLaplaceProblem (const unsigned int degree)
    :
    pf_degree (degree),
    pf_fe (FE_Q<dim>(pf_degree+1), dim,
        FE_Q<dim>(pf_degree), 1),
    pf_dof_handler (triangulation),
	
	vf_degree (degree),
	vf_fe (FE_Q<dim>(vf_degree), dim),
	vf_dof_handler (triangulation)
	
  {}




  template <int dim>
  void MixedLaplaceProblem<dim>::make_grid_and_dofs ()
  {
      
          std::vector<unsigned int> subdivisions (dim, 1);
          subdivisions[0] = 4;

          const Point<dim> bottom_left = (dim == 2 ?
                                          Point<dim>(data::left,data::bottom) :
                                          Point<dim>(-2,0,-1));
          const Point<dim> top_right   = (dim == 2 ?
                                          Point<dim>(data::right,data::top) :
                                          Point<dim>(0,1,0));

          GridGenerator::subdivided_hyper_rectangle (triangulation,
                                                     subdivisions,
                                                     bottom_left,
                                                     top_right);
      


      for (typename Triangulation<dim>::active_cell_iterator
           cell = triangulation.begin_active();
           cell != triangulation.end(); ++cell)
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
              if (cell->face(f)->center()[dim-1] == data::top)
                  cell->face(f)->set_all_boundary_ids(1);
              else if (cell->face(f)->center()[dim-1] == data::bottom)
                  cell->face(f)->set_all_boundary_ids(2);

     
    triangulation.refine_global (data::refinement_level);

    
    
        pf_dof_handler.distribute_dofs (pf_fe);
        DoFRenumbering::Cuthill_McKee (pf_dof_handler);
        std::vector<unsigned int> block_component (dim+1,0);
        block_component[dim] = 1;
        DoFRenumbering::component_wise (pf_dof_handler, block_component);
        
        pf_constraints.clear();
        pf_constraints.close();
        vf_dof_handler.distribute_dofs (vf_fe);
        
        
        std::vector<types::global_dof_index> dofs_per_block (2);
        DoFTools::count_dofs_per_block (pf_dof_handler, dofs_per_block, block_component);
        const unsigned int n_u = dofs_per_block[0],
                           n_pf = dofs_per_block[1],
						   n_vf = vf_dof_handler.n_dofs();
    
    // no refining mesh yet so not doing hanging node restraints - see step-31

//    USE THIS NUMBERING OF DOFS FOR RT SPACES
//    std::vector<types::global_dof_index> pf_dofs_per_component (dim+1);
//    DoFTools::count_dofs_per_component (pf_dof_handler, pf_dofs_per_component);
//    const unsigned int n_u = pf_dofs_per_component[0],
//                       n_pf = pf_dofs_per_component[dim],
//					   n_vf = vf_dof_handler.n_dofs();
    
      

    std::cout << "Problem Degree: "
    		  << data::problem_degree
              << std::endl
			  << "Refinement level: "
			  << data::refinement_level
			  << std::endl
			  << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: "
              << triangulation.n_cells()
              << std::endl
              << "Number of degrees of freedom: "
              << n_u + n_pf + n_vf
              << " (" << n_u << '+' << n_pf << '+' << n_vf << ')'
              << std::endl;
    
    ///////////////////////////////////////////////////////////////////////////////////
    
    
    pf_system_matrix.clear();
    BlockDynamicSparsityPattern dsp(2, 2);
    dsp.block(0, 0).reinit (n_u, n_u);
    dsp.block(1, 0).reinit (n_pf, n_u);
    dsp.block(0, 1).reinit (n_u, n_pf);
    dsp.block(1, 1).reinit (n_pf, n_pf);
    dsp.collect_sizes ();
    DoFTools::make_sparsity_pattern (pf_dof_handler, dsp, pf_constraints, false);
    
    pf_sparsity_pattern.copy_from(dsp);
    pf_system_matrix.reinit (pf_sparsity_pattern);
    
   		pf_solution.reinit (2);
        pf_solution.block(0).reinit (n_u);
        pf_solution.block(1).reinit (n_pf);
        pf_solution.collect_sizes ();

        pf_system_rhs.reinit (2);
        pf_system_rhs.block(0).reinit (n_u);
        pf_system_rhs.block(1).reinit (n_pf);
        pf_system_rhs.collect_sizes ();
          
    
    {
    	vf_system_matrix.clear ();
    	vf_constraints.clear ();
    	DoFTools::make_hanging_node_constraints (vf_dof_handler,
    	                                               vf_constraints);
    	vf_constraints.close();
     DynamicSparsityPattern dsp(vf_dof_handler.n_dofs());
     DoFTools::make_sparsity_pattern(vf_dof_handler,
                                        dsp,
                                        vf_constraints,
                                        /*keep_constrained_dofs = */ true);
        vf_sparsity_pattern.copy_from(dsp);
        
        vf_mass_matrix.reinit (vf_sparsity_pattern);
        vf_system_matrix.reinit (vf_sparsity_pattern);
        vf_solution.reinit (vf_dof_handler.n_dofs());
        vf_system_rhs.reinit (vf_dof_handler.n_dofs());
             
             
        MatrixCreator::create_mass_matrix(vf_dof_handler,
                                               QGauss<dim>(vf_fe.degree+2),
                                               vf_mass_matrix);
    }
    
    
  }



  template <int dim>
  void MixedLaplaceProblem<dim>::assemble_pf_system ()
  {
    QGauss<dim>   quadrature_formula(pf_degree+2);
    QGauss<dim-1> face_quadrature_formula(pf_degree+2);

    FEValues<dim> pf_fe_values (pf_fe, quadrature_formula,
                             update_values    | update_gradients |
                             update_quadrature_points  | update_JxW_values);
    FEFaceValues<dim> fe_face_values (pf_fe, face_quadrature_formula,
                                      update_values    | update_normal_vectors |
                                      update_quadrature_points  | update_JxW_values);

    const unsigned int   dofs_per_cell   = pf_fe.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();
    const unsigned int   n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    const RightHandSide<dim>          right_hand_side;
    const PressureBoundaryValues<dim> pressure_boundary_values;
    const KInverse<dim>               k_inverse;
    const K<dim>               		  k;

    std::vector<double> rhs_values (n_q_points);
    std::vector<double> boundary_values (n_face_q_points);
    std::vector<Tensor<2,dim> > k_inverse_values (n_q_points);
    std::vector<Tensor<2,dim> > k_values (n_q_points);

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    typename DoFHandler<dim>::active_cell_iterator
    cell = pf_dof_handler.begin_active(),
    endc = pf_dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        pf_fe_values.reinit (cell);
        local_matrix = 0;
        local_rhs = 0;

        right_hand_side.value_list (pf_fe_values.get_quadrature_points(),
                                    rhs_values);
        k_inverse.value_list (pf_fe_values.get_quadrature_points(),
                              k_inverse_values);
        k.value_list (pf_fe_values.get_quadrature_points(),
                                      k_values);

        for (unsigned int q=0; q<n_q_points; ++q)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              const Tensor<1,dim> phi_i_u     = pf_fe_values[velocities].value (i, q);
              const double        div_phi_i_u = pf_fe_values[velocities].divergence (i, q);
              const double        phi_i_p     = pf_fe_values[pressure].value (i, q);

              for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                  const Tensor<1,dim> phi_j_u     = pf_fe_values[velocities].value (j, q);
                  const double        div_phi_j_u = pf_fe_values[velocities].divergence (j, q);
                  const double        phi_j_p     = pf_fe_values[pressure].value (j, q);

                    local_matrix(i,j) += ( 1./data::lambda * phi_i_u * k_inverse_values[q] * phi_j_u
                                        - div_phi_i_u * phi_j_p
                                        - phi_i_p * div_phi_j_u)
                                       * pf_fe_values.JxW(q);
                }

              local_rhs(i) += phi_i_p *
                              rhs_values[q] *
                              pf_fe_values.JxW(q);
              // BE CAREFUL HERE ONCE K is not constant or 1 anymore
            }


        for (unsigned int face_no=0;
             face_no<GeometryInfo<dim>::faces_per_cell;
             ++face_no)
          if (cell->face(face_no)->at_boundary()
                  &&
                  (cell->face(face_no)->boundary_id() == 1)) // Basin top has boundary id 1
            {
              fe_face_values.reinit (cell, face_no);
                // DIRICHLET CONDITION FOR TOP. pf = pr at top of basin 

              pressure_boundary_values
              .value_list (fe_face_values.get_quadrature_points(),
                           boundary_values);

              for (unsigned int q=0; q<n_face_q_points; ++q)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                  local_rhs(i) += -( fe_face_values[velocities].value (i, q) *
                                    fe_face_values.normal_vector(q) *
                                    boundary_values[q] *
                                    fe_face_values.JxW(q));  
            }


        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            pf_system_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               local_matrix(i,j));
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          pf_system_rhs(local_dof_indices[i]) += local_rhs(i);
      }
    
    
 // NOW NEED TO STRONGLY IMPOSE THE FLUX CONDITIONS BOTH ON SIDES AND BOTTOM
    
    std::map<types::global_dof_index, double> boundary_values_flux; 
    {
            types::global_dof_index n_dofs = pf_dof_handler.n_dofs();
            std::vector<bool> componentVector(dim + 1, true); // condition is on pressue
            // setting flux value for the sides at 0 ON THE PRESSURE
            componentVector[dim] = false;
            std::vector<bool> selected_dofs(n_dofs);
            std::set< types::boundary_id > boundary_ids;
            boundary_ids.insert(0);
        
            DoFTools::extract_boundary_dofs(pf_dof_handler, ComponentMask(componentVector),
                    selected_dofs, boundary_ids);

            for (types::global_dof_index i = 0; i < n_dofs; i++) {
                if (selected_dofs[i]) boundary_values_flux[i] = 0.0; // Side boudaries have flux 0 on pressure
            }

    }

      // Apply the conditions
      
    MatrixTools::apply_boundary_values(boundary_values_flux,
            pf_system_matrix, pf_solution, pf_system_rhs);
      
     
  }
  

  template <int dim>
  void MixedLaplaceProblem<dim>::assemble_vf_system ()
  {
	
	  QGauss<dim>   quadrature_formula(vf_degree+2);

	     FEValues<dim> vf_fe_values (vf_fe, quadrature_formula,
	                              update_values    | update_gradients |
	                              update_quadrature_points  | update_JxW_values);
	     FEValues<dim> pf_fe_values (pf_fe, quadrature_formula,
	                              update_values    | update_gradients );	       
	       
	     const unsigned int   dofs_per_cell   = vf_fe.dofs_per_cell;
	     const unsigned int   n_q_points      = quadrature_formula.size();

	     FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
	     Vector<double>       local_rhs (dofs_per_cell);
	     std::vector<Tensor<1,dim> > grad_pf_values (n_q_points);

	     std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
	       
	     
	       std::vector<Tensor<1,dim>>     negunitz_values (n_q_points);
	       std::vector<Tensor<1,dim>>     rock_vel_values (n_q_points);
	       // std::vector<Tensor<1,dim>>     pressure_flux_values (n_q_points);
	       
	       const FEValuesExtractors::Vector velocities (0);
	       const FEValuesExtractors::Scalar pressure (dim);


	     typename DoFHandler<dim>::active_cell_iterator
	     cell = vf_dof_handler.begin_active(),
	     endc = vf_dof_handler.end();
//	     typename DoFHandler<dim>::active_cell_iterator
//		 pf_cell = pf_dof_handler.begin_active();
		 
	     for (; cell!=endc; ++cell/*, ++pf_cell*/)
	     	 {
	          for (unsigned int i=0; i<dofs_per_cell; ++i)
	          {
	              const unsigned int
	              component_i = vf_fe.system_to_component_index(i).first;
	              
	              for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
	                local_rhs(i) += 1*vf_fe_values.shape_value(i,q_point)*vf_fe_values.JxW(q_point);
	              
	          }

	        cell->get_dof_indices (local_dof_indices);
	        for (unsigned int i=0; i<dofs_per_cell; ++i)
	          for (unsigned int j=0; j<dofs_per_cell; ++j)
	            vf_system_matrix.add (local_dof_indices[i],
	                               local_dof_indices[j],
	                               local_matrix(i,j));
	        
	        for (unsigned int i=0; i<dofs_per_cell; ++i)
	          vf_system_rhs(local_dof_indices[i]) += local_rhs(i);
	     }
	  
  }




  template <int dim>
  void MixedLaplaceProblem<dim>::solve ()
  {
	  std::cout << "   Solving for p_f..." << std::endl;
	  
      SparseDirectUMFPACK  pf_direct;
      pf_direct.initialize(pf_system_matrix);
      pf_direct.vmult (pf_solution, pf_system_rhs);
      
      assemble_vf_system ();
      vf_constraints.condense (vf_system_matrix);
      vf_constraints.condense (vf_system_rhs);
      
	  std::cout << "   Solving for v_f..." << std::endl;
	  

      
      vf_system_matrix.copy_from(vf_mass_matrix);      
      SparseDirectUMFPACK  vf_direct;
      vf_direct.initialize(vf_system_matrix);
      vf_direct.vmult (vf_solution, vf_system_rhs);
      
      

  }
    

  template <int dim>
  void MixedLaplaceProblem<dim>::compute_errors () const
  {
    const ComponentSelectFunction<dim>
    pressure_mask (dim, dim+1);
    const ComponentSelectFunction<dim>
    velocity_mask(std::make_pair(0, dim), dim+1);

    ExactSolution_pf<dim> exact_solution_pf;
    Vector<double> cellwise_errors_pf (triangulation.n_active_cells());

    QTrapez<1>     q_trapez;
    QIterated<dim> quadrature (q_trapez, vf_degree+2);

    VectorTools::integrate_difference (pf_dof_handler, pf_solution, exact_solution_pf,
                                       cellwise_errors_pf, quadrature,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    const double pf_l2_error = cellwise_errors_pf.l2_norm();

    VectorTools::integrate_difference (pf_dof_handler, pf_solution, exact_solution_pf,
                                       cellwise_errors_pf, quadrature,
                                       VectorTools::L2_norm,
                                       &velocity_mask);
    const double u_l2_error = cellwise_errors_pf.l2_norm();

	    ExactSolution_vf<dim> exact_solution_vf;
	    Vector<double> cellwise_errors_vf (triangulation.n_active_cells());


	    VectorTools::integrate_difference (vf_dof_handler, vf_solution, exact_solution_vf,
	                                       cellwise_errors_vf, quadrature,
	                                       VectorTools::L2_norm);
	    const double vf_l2_error = cellwise_errors_vf.l2_norm();

    std::cout << "Errors: ||e_pf||_L2 = " << pf_l2_error 
              << "," << std::endl <<   "        ||e_u||_L2 = " << u_l2_error
			  << ", " << std::endl <<   "        ||e_vf||_L2 = " << vf_l2_error
			  << ". " << std::endl;


  }


  template <int dim>
  void MixedLaplaceProblem<dim>::output_results () const
  {
//      
//      std::vector<std::string> solution_names;
//    switch (dim)
//      {
//      case 2:
//        solution_names.push_back ("u");
//        solution_names.push_back ("v");
//        solution_names.push_back ("p");
//        break;
//
//      case 3:
//        solution_names.push_back ("u");
//        solution_names.push_back ("v");
//        solution_names.push_back ("w");
//        solution_names.push_back ("p");
//        break;
//
//      default:
//        Assert (false, ExcNotImplemented());
//      }
//
//
//    DataOut<dim> data_out;
//
//    data_out.attach_dof_handler (pf_dof_handler);
//    data_out.add_data_vector (pf_solution, solution_names);
//
//
//    data_out.build_patches ();
//
//    std::ofstream output ("pf_solution.vtk");
//    data_out.write_vtk (output);
//    
    
    

    std::vector<std::string> pf_names (dim, "uf");
    pf_names.push_back ("p_f");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    pf_component_interpretation
    (dim+1, DataComponentInterpretation::component_is_scalar);
    for (unsigned int i=0; i<dim; ++i)
      pf_component_interpretation[i]
        = DataComponentInterpretation::component_is_part_of_vector;
    DataOut<dim> data_out;
    
    data_out.add_data_vector (pf_dof_handler, pf_solution,
                              pf_names, pf_component_interpretation);
    data_out.add_data_vector (vf_dof_handler, vf_solution,
                              "v_f");
    
    data_out.build_patches (std::min(pf_degree, vf_degree));
    std::ostringstream filename;
    filename << "solution.vtk";
    std::ofstream output (filename.str().c_str());
    data_out.write_vtk (output);
  }
  
  




  template <int dim>
  void MixedLaplaceProblem<dim>::run ()
  {
   make_grid_and_dofs();
   assemble_pf_system ();
   solve ();
   compute_errors ();
   output_results ();
    
  }
}



int main ()
{
  try
    {
      using namespace dealii;
      using namespace Step20;

      MixedLaplaceProblem<data::dimension> mixed_laplace_problem(data::problem_degree);
      mixed_laplace_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
